# macos-video-auto-ocr-ass

## 介紹

本專案提供多種自動化字幕（ASS）翻譯工具，支援本地大型語言模型（llama.cpp GGUF）與 Hugging Face MarianMT 神經機器翻譯模型，適合批次處理影片字幕、保留 ASS 格式標籤、支援多語言翻譯。

---

## 依賴安裝（建議使用 Poetry）

### 1. 安裝 Poetry（如尚未安裝）

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. 安裝專案依賴

```bash
poetry install
```

### 3. 進入虛擬環境

```bash
poetry shell
```

### 4. 若需 Apple Silicon/M1/M2/M3 GPU 加速，請在 poetry shell 內安裝最新版 torch：

```bash
pip install --upgrade torch torchvision torchaudio
```

---

## 開發工具（Makefile）

本專案提供 Makefile 來簡化開發和測試流程。

### 查看所有可用命令

```bash
make help
```

### 測試相關命令

```bash
# 運行所有測試
make test

# 運行單元測試
make test-unit

# 運行整合測試
make test-integration

# 運行快速測試（排除慢速測試）
make test-fast

# 運行 macOS 特定測試
make test-macos

# 運行測試並生成覆蓋率報告
make test-coverage

# 運行特定測試檔案
make test-specific FILE=test_config.py

# 運行特定測試函數
make test-specific FILE=test_config.py FUNCTION=TestAppConfig
```

### 代碼品質檢查

```bash
# 格式化代碼
make format

# 檢查代碼風格
make lint

# 類型檢查
make type-check

# 運行所有代碼品質檢查
make all-checks
```

### 其他開發命令

```bash
# 安裝開發依賴
make install-dev

# 清理暫存檔案
make clean

# 運行所有測試和代碼品質檢查
make all-tests

# 啟動 Poetry shell
make shell

# 檢查環境
make check-env
```

### 快速開發設置

```bash
# 一次性設置開發環境
make dev
```

---

## 1. 使用 llama-cpp-python + GGUF 模型

### 指令
```bash
python translate_ass_llama.py input.ass output.ass [--src-lang 英文名] --tgt-lang "Traditional Chinese" [--show-text]
```

### 主要參數
- `input.ass`：輸入 ASS 字幕檔
- `output.ass`：輸出 ASS 字幕檔
- `--src-lang`：來源語言（預設 auto，自動偵測）
- `--tgt-lang`：目標語言（如 "Traditional Chinese"、"Japanese" 等，建議用英文全名）
- `--show-text`：顯示翻譯前後內容
- `--model-repo`、`--model-filename`：Hugging Face GGUF 模型 repo 與檔名（預設 X-ALMA-13B-Group6-GGUF）
- `--model-dir`：模型下載資料夾

### 範例
```bash
python translate_ass_llama.py input.ass output.ass --tgt-lang "Traditional Chinese" --show-text
```

### 說明
- 會自動從 Hugging Face 下載 GGUF 模型
- 目標語言為 "Traditional Chinese" 時，會自動用 opencc 轉換為正體中文
- 保留 ASS tag（如 \pos、\move）
- 同一時間出現的字幕會合併翻譯

---

## 2. 使用 MarianMT (Hugging Face Transformers)

### 指令
```bash
python translate_ass_marianmt.py input.ass output.ass --src-lang en --tgt-lang zht [--show-text]
```

### 主要參數
- `input.ass`：輸入 ASS 字幕檔
- `output.ass`：輸出 ASS 字幕檔
- `--src-lang`：來源語言（ISO 代碼，如 en, ja, zh）
- `--tgt-lang`：zht=繁體、zhs=簡體、zh=繁體（全部都用 zh 模型，zht/zh 會自動轉繁體）
- `--model`：自訂 Hugging Face MarianMT 模型名稱（預設自動推斷）
- `--device`：運算裝置（如 cpu, mps, cuda）
- `--show-text`：顯示翻譯前後內容

### 範例
```bash
python translate_ass_marianmt.py input.ass output.ass --src-lang en --tgt-lang zht --show-text
```

### 說明
- 會自動推斷並下載 MarianMT 模型
- `--tgt-lang zht` 或 `zh` 會自動用 opencc 轉換為繁體中文
- `--tgt-lang zhs` 直接用簡體中文
- 保留 ASS tag，合併同時段字幕翻譯

---

## 常見問題

### Q1. 為什麼翻譯結果有時不是目標語言？
- 請確認 `--tgt-lang` 參數正確，並盡量用明確的語言名稱（llama）或 ISO 代碼（MarianMT）。
- 若用 LLM，建議加強 prompt 或每行單獨翻譯。

### Q2. 如何加速翻譯？
- Apple Silicon/M1/M2/M3 建議用 `--device mps`。
- PC/NVIDIA GPU 可用 `--device cuda`。

### Q3. opencc 沒有作用？
- 請確認已安裝 opencc：
  ```bash
  poetry add opencc
  ```
  或
  ```bash
  pip install opencc
  ```
- 只有 `--tgt-lang` 為 zht/zh 時才會自動轉繁體。

### Q4. 模型下載很慢？
- 可先手動下載 Hugging Face 模型，放到指定資料夾。

---

## 參考連結
- [Poetry 官方網站](https://python-poetry.org/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Helsinki-NLP/MarianMT](https://huggingface.co/Helsinki-NLP)
- [X-ALMA-13B-Group6-GGUF](https://huggingface.co/mradermacher/X-ALMA-13B-Group6-GGUF)
- [pysubs2](https://github.com/pympi/PySubs2)
- [opencc](https://github.com/BYVoid/OpenCC)
