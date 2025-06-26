# 測試文檔

本目錄包含 `macos-video-auto-ocr-ass` 專案的完整測試套件。

## 測試結構

```
tests/
├── conftest.py              # pytest 配置和共用夾具
├── test_config.py           # 配置模組測試
├── test_ass_utils.py        # ASS 工具模組測試
├── test_logger.py           # 日誌系統測試
├── test_translators.py      # 翻譯器模組測試
├── test_video_utils.py      # 影片工具模組測試
├── run_tests.py             # 測試運行腳本
└── README.md               # 本文件
```

## 安裝測試依賴

```bash
# 安裝開發依賴
poetry install --with dev

# 或者手動安裝
pip install pytest pytest-cov pytest-mock pytest-xdist
```

## 運行測試

### 使用 pytest 直接運行

```bash
# 運行所有測試
pytest

# 運行特定測試檔案
pytest tests/test_config.py

# 運行特定測試類別
pytest tests/test_config.py::TestAppConfig

# 運行特定測試函數
pytest tests/test_config.py::TestAppConfig::test_app_config_defaults

# 生成覆蓋率報告
pytest --cov=macos_video_auto_ocr_ass --cov-report=html
```

### 使用測試運行腳本

```bash
# 運行所有測試
python tests/run_tests.py all

# 運行單元測試
python tests/run_tests.py unit

# 運行整合測試
python tests/run_tests.py integration

# 運行快速測試（排除慢速測試）
python tests/run_tests.py fast

# 運行 macOS 特定測試
python tests/run_tests.py macos

# 生成覆蓋率報告
python tests/run_tests.py all --coverage

# 詳細輸出
python tests/run_tests.py all --verbose

# 運行特定測試檔案
python tests/run_tests.py specific test_config.py

# 運行特定測試類別
python tests/run_tests.py specific test_config.py TestAppConfig
```

## 測試標記

測試使用以下標記進行分類：

- `@pytest.mark.unit` - 單元測試
- `@pytest.mark.integration` - 整合測試
- `@pytest.mark.slow` - 慢速測試
- `@pytest.mark.macos` - 需要 macOS 系統的測試
- `@pytest.mark.vision` - 需要 Vision 框架的測試

### 運行特定類型的測試

```bash
# 只運行單元測試
pytest -m unit

# 只運行整合測試
pytest -m integration

# 排除慢速測試
pytest -m "not slow"

# 只運行 macOS 測試
pytest -m macos
```

## 測試覆蓋率

### 生成覆蓋率報告

```bash
# 生成終端覆蓋率報告
pytest --cov=macos_video_auto_ocr_ass

# 生成 HTML 覆蓋率報告
pytest --cov=macos_video_auto_ocr_ass --cov-report=html

# 生成多種格式的覆蓋率報告
pytest --cov=macos_video_auto_ocr_ass --cov-report=html --cov-report=term --cov-report=xml
```

HTML 報告會生成在 `htmlcov/` 目錄中，可以用瀏覽器打開 `htmlcov/index.html` 查看詳細報告。

## 測試配置

### pytest.ini

主要的 pytest 配置在 `pytest.ini` 文件中：

- 測試路徑：`tests/`
- 測試檔案模式：`test_*.py`
- 測試類別模式：`Test*`
- 測試函數模式：`test_*`
- 預設選項：詳細輸出、短回溯、彩色輸出等

### conftest.py

包含共用的測試夾具：

- `temp_dir` - 臨時目錄夾具
- `sample_image` - 測試用圖像
- `sample_ass_content` - 測試用 ASS 字幕內容
- `sample_ass_file` - 測試用 ASS 檔案
- `app_config` - 測試用應用程式配置
- `mock_video_path` - 模擬影片路徑
- `mock_ocr_results` - 模擬 OCR 結果

## 測試最佳實踐

### 1. 測試命名

- 測試檔案：`test_<module_name>.py`
- 測試類別：`Test<ClassName>`
- 測試函數：`test_<function_name>_<scenario>`

### 2. 測試組織

- 每個模組都有對應的測試檔案
- 測試類別按功能分組
- 使用描述性的測試函數名稱

### 3. 模擬和夾具

- 使用 `unittest.mock` 模擬外部依賴
- 使用 pytest 夾具提供測試數據
- macOS 特定功能使用 `@pytest.mark.skipif` 跳過

### 4. 邊界情況

- 測試空輸入
- 測試無效輸入
- 測試異常情況
- 測試極限值

## 持續整合

### GitHub Actions

可以在 `.github/workflows/tests.yml` 中配置自動測試：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      - name: Run tests
        run: poetry run pytest
```

## 故障排除

### 常見問題

1. **ImportError: No module named 'macos_video_auto_ocr_ass'**
   - 確保在專案根目錄運行測試
   - 確保已安裝開發依賴

2. **macOS 特定測試失敗**
   - 這些測試只在 macOS 上運行
   - 在其他系統上會自動跳過

3. **Vision 框架測試失敗**
   - 需要 macOS 系統
   - 需要安裝 pyobjc 依賴

### 調試測試

```bash
# 詳細輸出
pytest -v

# 顯示本地變數
pytest -l

# 在失敗時進入 pdb
pytest --pdb

# 顯示最慢的測試
pytest --durations=10
```

## 貢獻指南

添加新測試時：

1. 創建對應的測試檔案
2. 使用適當的測試標記
3. 添加必要的模擬
4. 確保測試覆蓋率
5. 更新本文檔

## 相關文檔

- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)
- [unittest.mock 文檔](https://docs.python.org/3/library/unittest.mock.html) 