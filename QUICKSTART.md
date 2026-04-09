# Quickstart: запуск переводчика локально

Инструмент запускается **локально** из вашего репозитория и не требует публикации.

## 1) Проверка среды
```bash
python3 --version
python3 -m py_compile precision_translator.py
```

## 2) Самый простой запуск (без API, через stub)
```bash
./run_translator.sh stub zh ru technical '设备温度必须保持在25°C，压力不超过2.5MPa。' --json
```

или напрямую:
```bash
python3 precision_translator.py \
  --provider stub \
  --source-lang zh \
  --target-lang ru \
  --domain technical \
  --text '设备温度必须保持在25°C，压力不超过2.5MPa。' \
  --json
```

## 3) Запуск с реальным ИИ (OpenAI-compatible API)
```bash
export OPENAI_API_KEY='ваш_ключ'
python3 precision_translator.py \
  --provider openai-compatible \
  --base-url 'https://api.openai.com/v1' \
  --model 'gpt-4.1' \
  --source-lang zh \
  --target-lang ru \
  --domain conversational \
  --text '你好，我们明天确认合同。' \
  --json
```

## 3.1) Запуск через AgentPlatform (по вашему curl)
1. Положите ключ **в переменную окружения**, а не в чат:
```bash
export AGENTPLATFORM_KEY='ваш_секретный_ключ'
```
2. Запуск:
```bash
python3 precision_translator.py \
  --provider agentplatform \
  --source-lang zh \
  --target-lang ru \
  --domain technical \
  --text '设备温度必须保持在25°C，压力不超过2.5MPa。' \
  --json
```
3. Опционально (чтобы не писать `export` каждый раз): добавьте строку `export AGENTPLATFORM_KEY='...'` в `~/.bashrc`, затем `source ~/.bashrc`.

## 4) Если хотите увидеть справку по всем флагам
```bash
python3 precision_translator.py --help
```

## 5) Частые ошибки
- `Environment variable OPENAI_API_KEY is empty` → задайте ключ или используйте `--provider stub`.
- `Environment variable AGENTPLATFORM_KEY is empty` → задайте ключ: `export AGENTPLATFORM_KEY='...'`.
- Одинаковые `--source-lang` и `--target-lang` → должны отличаться (`zh` и `ru`).
