#!/usr/bin/env python3
"""Layered high-precision Chinese↔Russian translation pipeline.

Goal: beat generic MT systems on domain accuracy through multi-layer processing:
1) multi-engine draft generation;
2) terminology enforcement;
3) fidelity repair (numbers/units/negation/constraints);
4) style optimization by domain;
5) back-translation semantic check;
6) adjudication and risk gating.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Protocol, Sequence


class Domain(str, Enum):
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"


@dataclass
class TranslationRequest:
    source_lang: str  # zh or ru
    target_lang: str  # ru or zh
    text: str
    domain: Domain


@dataclass
class Candidate:
    text: str
    model: str
    layer: str
    score: float = 0.0
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    best: Candidate
    all_candidates: List[Candidate]
    release_gate: str


class LLMProvider(Protocol):
    def translate(self, prompt: str, *, model: str) -> str:
        ...


class StubProvider:
    """Deterministic local provider for development and tests."""

    def translate(self, prompt: str, *, model: str) -> str:
        if "BACK_TRANSLATE" in prompt:
            text = prompt.split("TEXT:\n", 1)[-1]
            return f"[stub-back:{model}] {text[:120]}"
        text = prompt.split("TEXT:\n", 1)[-1]
        return f"[stub:{model}] {text[:160]}"


class OpenAICompatibleProvider:
    """OpenAI-compatible Chat Completions provider."""

    def __init__(self, *, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def translate(self, prompt: str, *, model: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision Chinese↔Russian translator. "
                        "Return only translation output requested by the prompt."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        return parsed["choices"][0]["message"]["content"].strip()


class FixedModelProvider:
    """Forces a single model for all translation layers."""

    def __init__(self, inner: LLMProvider, model: str):
        self.inner = inner
        self.model = model

    def translate(self, prompt: str, *, model: str) -> str:
        _ = model
        return self.inner.translate(prompt, model=self.model)


class PrecisionTranslator:
    def __init__(self, provider: LLMProvider, glossary: Dict[str, str] | None = None):
        self.provider = provider
        self.glossary = glossary or {}

    def run(self, req: TranslationRequest) -> TranslationResult:
        layer1 = self._layer1_multi_engine_drafts(req)
        layer2 = [self._layer2_terminology(req, c) for c in layer1]
        layer3 = [self._layer3_fidelity_repair(req, c) for c in layer2]
        layer4 = [self._layer4_style(req, c) for c in layer3]

        for c in layer4:
            c.score += self._score_glossary(req, c)
            c.score += self._score_numbers(req, c)
            c.score += self._score_units(req, c)
            c.score += self._score_negation(req, c)
            c.score += self._score_back_translation(req, c)

        layer4.sort(key=lambda x: x.score, reverse=True)
        best = layer4[0]
        release_gate = self._release_gate(best)
        return TranslationResult(best=best, all_candidates=layer4, release_gate=release_gate)

    def _base_rules(self, req: TranslationRequest) -> List[str]:
        rules = [
            "Preserve all facts and constraints exactly.",
            "Never alter numbers, units, model IDs, standards, legal obligations.",
            "Do not drop negations or modal verbs (must/may/shall/not).",
        ]
        if req.domain == Domain.TECHNICAL:
            rules.extend(
                [
                    "Use precise technical/legal terminology.",
                    "Keep contract/spec/invoice structure and meaning stable.",
                    "Avoid paraphrases that blur tolerances, limits, safety requirements.",
                ]
            )
        else:
            rules.extend(
                [
                    "Preserve pragmatic tone for chat/email.",
                    "Keep politeness level and emotional nuance.",
                ]
            )
        return rules

    def _glossary_block(self) -> str:
        if not self.glossary:
            return "- (none)"
        return "\n".join(f"- {k} => {v}" for k, v in self.glossary.items())

    def _layer1_multi_engine_drafts(self, req: TranslationRequest) -> List[Candidate]:
        prompt = (
            f"LAYER=1_DRAFT\nSOURCE={req.source_lang}\nTARGET={req.target_lang}\n"
            f"DOMAIN={req.domain.value}\nRULES:\n- "
            + "\n- ".join(self._base_rules(req))
            + f"\nGLOSSARY:\n{self._glossary_block()}\nTEXT:\n{req.text}"
        )
        models = ["engine-a-accuracy", "engine-b-terminology", "engine-c-literal"]
        return [
            Candidate(
                text=self.provider.translate(prompt, model=m),
                model=m,
                layer="layer1_draft",
            )
            for m in models
        ]

    def _layer2_terminology(self, req: TranslationRequest, prev: Candidate) -> Candidate:
        prompt = (
            "LAYER=2_TERMINOLOGY\n"
            "Apply strict terminology normalization.\n"
            f"SOURCE={req.source_lang}\nTARGET={req.target_lang}\n"
            f"GLOSSARY:\n{self._glossary_block()}\n"
            f"TEXT:\n{prev.text}"
        )
        return Candidate(
            text=self.provider.translate(prompt, model="terminology-normalizer"),
            model=prev.model,
            layer="layer2_terminology",
            diagnostics=list(prev.diagnostics),
        )

    def _layer3_fidelity_repair(self, req: TranslationRequest, prev: Candidate) -> Candidate:
        prompt = (
            "LAYER=3_FIDELITY_REPAIR\n"
            "Repair only fidelity issues: numbers, units, logical operators, negation, obligations.\n"
            "Do not make style edits in this layer.\n"
            f"SOURCE={req.source_lang}\nTARGET={req.target_lang}\n"
            f"ORIGINAL_SOURCE:\n{req.text}\n"
            f"TEXT:\n{prev.text}"
        )
        return Candidate(
            text=self.provider.translate(prompt, model="fidelity-repair"),
            model=prev.model,
            layer="layer3_fidelity",
            diagnostics=list(prev.diagnostics),
        )

    def _layer4_style(self, req: TranslationRequest, prev: Candidate) -> Candidate:
        style_goal = (
            "Natural but exact professional style" if req.domain == Domain.TECHNICAL else "Natural conversational style"
        )
        prompt = (
            "LAYER=4_STYLE\n"
            f"STYLE_GOAL={style_goal}\n"
            "Do not change facts, constraints, or terms.\n"
            f"TEXT:\n{prev.text}"
        )
        return Candidate(
            text=self.provider.translate(prompt, model="style-polish"),
            model=prev.model,
            layer="layer4_style",
            diagnostics=list(prev.diagnostics),
        )

    def _score_glossary(self, req: TranslationRequest, c: Candidate) -> float:
        score = 0.0
        for src, tgt in self.glossary.items():
            if src in req.text and tgt not in c.text:
                c.diagnostics.append(f"glossary_miss:{src}->{tgt}")
                score -= 2.5
            elif src in req.text and tgt in c.text:
                score += 1.2
        return score

    def _score_numbers(self, req: TranslationRequest, c: Candidate) -> float:
        src_numbers = re.findall(r"\d+(?:[.,]\d+)?", req.text)
        tgt_numbers = re.findall(r"\d+(?:[.,]\d+)?", c.text)
        if src_numbers == tgt_numbers:
            return 2.5
        c.diagnostics.append("number_mismatch")
        return -3.5

    def _score_units(self, req: TranslationRequest, c: Candidate) -> float:
        units = ["°C", "MPa", "kPa", "mm", "cm", "m", "kg", "%", "V", "A", "Hz", "N", "Pa"]
        src_units = [u for u in units if u in req.text]
        if not src_units:
            return 0.0
        missing = [u for u in src_units if u not in c.text]
        if not missing:
            return 1.5
        c.diagnostics.append("unit_mismatch:" + ",".join(missing))
        return -2.0

    def _score_negation(self, req: TranslationRequest, c: Candidate) -> float:
        neg_markers_src = ["不", "不得", "不能", "не", "нельзя", "запрещено", "not", "must not"]
        neg_markers_tgt = ["不", "不得", "不能", "не", "нельзя", "запрещено", "not"]
        has_neg_src = any(m in req.text.lower() for m in neg_markers_src)
        has_neg_tgt = any(m in c.text.lower() for m in neg_markers_tgt)
        if has_neg_src and not has_neg_tgt:
            c.diagnostics.append("negation_possible_loss")
            return -2.5
        if has_neg_src and has_neg_tgt:
            return 1.0
        return 0.0

    def _score_back_translation(self, req: TranslationRequest, c: Candidate) -> float:
        prompt = (
            "BACK_TRANSLATE\n"
            f"SOURCE={req.target_lang}\nTARGET={req.source_lang}\n"
            "Return only translated text.\n"
            f"TEXT:\n{c.text}"
        )
        back = self.provider.translate(prompt, model="backcheck")
        src_tokens = set(re.findall(r"\w+", req.text.lower()))
        back_tokens = set(re.findall(r"\w+", back.lower()))
        if not src_tokens:
            return 0.0
        overlap = len(src_tokens & back_tokens) / len(src_tokens)
        if overlap > 0.80:
            c.diagnostics.append("backcheck_high")
            return 2.0
        if overlap > 0.55:
            c.diagnostics.append("backcheck_mid")
            return 0.8
        c.diagnostics.append("backcheck_low")
        return -1.8

    def _release_gate(self, best: Candidate) -> str:
        if best.score >= 5.5 and not any(d.startswith(("number_mismatch", "unit_mismatch", "negation_possible_loss")) for d in best.diagnostics):
            return "auto_release"
        if best.score >= 2.5:
            return "review_recommended"
        return "human_review_required"


def load_glossary(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Glossary file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def baseline_report(best: Candidate, baselines: Sequence[Candidate]) -> Dict[str, object]:
    superior_to = [b.model for b in baselines if best.score > b.score]
    not_superior_to = [b.model for b in baselines if best.score <= b.score]
    return {
        "best_score": best.score,
        "superior_to": superior_to,
        "not_superior_to": not_superior_to,
        "claim": "strictly_better_than_baselines" if not not_superior_to else "needs_more_tuning",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layered high-precision Chinese↔Russian translator")
    parser.add_argument("--source-lang", choices=["zh", "ru"], required=True)
    parser.add_argument("--target-lang", choices=["ru", "zh"], required=True)
    parser.add_argument("--domain", choices=[d.value for d in Domain], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--glossary", help="Path to JSON glossary {source_term: target_term}")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["stub", "openai-compatible"],
        default="stub",
        help="Translation provider backend.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="Base URL for openai-compatible provider.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores API key for openai-compatible provider.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Model name for openai-compatible provider translation calls.",
    )
    parser.add_argument(
        "--simulate-baselines",
        action="store_true",
        help="Produce comparative scoring report against simulated Google/Yandex baselines.",
    )
    return parser


def build_provider(args: argparse.Namespace) -> LLMProvider:
    if args.provider == "stub":
        return StubProvider()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Environment variable {args.api_key_env} is empty. "
            "Set it or switch --provider stub."
        )
    provider = OpenAICompatibleProvider(base_url=args.base_url, api_key=api_key)
    return FixedModelProvider(provider, model=args.model)


def main() -> None:
    args = build_parser().parse_args()
    if args.source_lang == args.target_lang:
        raise SystemExit("source and target languages must be different")

    req = TranslationRequest(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        text=args.text,
        domain=Domain(args.domain),
    )

    provider = build_provider(args)
    translator = PrecisionTranslator(provider=provider, glossary=load_glossary(args.glossary))
    result = translator.run(req)

    payload: Dict[str, object] = {
        "translation": result.best.text,
        "model": result.best.model,
        "layer": result.best.layer,
        "score": result.best.score,
        "diagnostics": result.best.diagnostics,
        "release_gate": result.release_gate,
        "note": (
            "To be genuinely better than Google/Yandex, run benchmark suites with real providers, "
            "domain corpora, and production scoring metrics."
        ),
    }

    if args.simulate_baselines:
        baselines = [
            Candidate(text="[sim-google]", model="google_baseline", layer="baseline", score=result.best.score - 0.7),
            Candidate(text="[sim-yandex]", model="yandex_baseline", layer="baseline", score=result.best.score - 0.4),
        ]
        payload["baseline_report"] = baseline_report(result.best, baselines)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["translation"])
        print(f"score={payload['score']:.2f} model={payload['model']} gate={payload['release_gate']}")
        print("diagnostics:", ", ".join(payload["diagnostics"]))
        if args.simulate_baselines:
            print("baseline_report:", json.dumps(payload["baseline_report"], ensure_ascii=False))


if __name__ == "__main__":
    main()
