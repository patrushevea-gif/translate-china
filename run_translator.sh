#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for precision_translator.py
# Usage examples:
#   ./run_translator.sh stub zh ru technical '设备温度必须保持在25°C，压力不超过2.5MPa。'
#   OPENAI_API_KEY=... ./run_translator.sh openai-compatible zh ru conversational '你好，明天见。'

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <provider> <source_lang> <target_lang> <domain> <text> [extra_args...]"
  echo "provider: stub | openai-compatible"
  echo "source_lang/target_lang: zh | ru"
  echo "domain: conversational | technical"
  exit 1
fi

PROVIDER="$1"
SOURCE_LANG="$2"
TARGET_LANG="$3"
DOMAIN="$4"
TEXT="$5"
shift 5

python3 precision_translator.py \
  --provider "$PROVIDER" \
  --source-lang "$SOURCE_LANG" \
  --target-lang "$TARGET_LANG" \
  --domain "$DOMAIN" \
  --text "$TEXT" \
  "$@"
