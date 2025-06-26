# macOS 影片自動 OCR 字幕工具 Makefile
# 提供便捷的開發和測試命令

.PHONY: help install install-dev test test-unit test-integration test-fast test-macos test-coverage test-specific clean format lint type-check all-tests

# 預設目標
help:
	@echo "macOS 影片自動 OCR 字幕工具 - 可用命令："
	@echo ""
	@echo "📦 安裝相關："
	@echo "  install      - 安裝專案依賴"
	@echo "  install-dev  - 安裝開發依賴"
	@echo ""
	@echo "🧪 測試相關："
	@echo "  test         - 運行所有測試"
	@echo "  test-unit    - 運行單元測試"
	@echo "  test-integration - 運行整合測試"
	@echo "  test-fast    - 運行快速測試（排除慢速測試）"
	@echo "  test-macos   - 運行 macOS 特定測試"
	@echo "  test-coverage - 運行測試並生成覆蓋率報告"
	@echo "  test-specific - 運行特定測試檔案"
	@echo ""
	@echo "🔧 代碼品質："
	@echo "  format       - 格式化代碼（black + isort）"
	@echo "  lint         - 檢查代碼風格（flake8）"
	@echo "  type-check   - 類型檢查（mypy）"
	@echo "  all-checks   - 運行所有代碼品質檢查"
	@echo ""
	@echo "🧹 清理："
	@echo "  clean        - 清理暫存檔案"
	@echo ""
	@echo "📊 完整檢查："
	@echo "  all-tests    - 運行所有測試和代碼品質檢查"

# 安裝相關
install:
	@echo "📦 安裝專案依賴..."
	poetry install --only main

install-dev:
	@echo "📦 安裝開發依賴..."
	poetry install

# 測試相關
test:
	@echo "🧪 運行所有測試..."
	poetry run python tests/run_tests.py all

test-unit:
	@echo "🧪 運行單元測試..."
	poetry run python tests/run_tests.py unit

test-integration:
	@echo "🧪 運行整合測試..."
	poetry run python tests/run_tests.py integration

test-fast:
	@echo "🧪 運行快速測試..."
	poetry run python tests/run_tests.py fast

test-macos:
	@echo "🧪 運行 macOS 特定測試..."
	poetry run python tests/run_tests.py macos

test-coverage:
	@echo "🧪 運行測試並生成覆蓋率報告..."
	poetry run python tests/run_tests.py all --coverage

test-specific:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ 請指定測試檔案，例如：make test-specific FILE=test_config.py"; \
		exit 1; \
	fi
	@echo "🧪 運行特定測試：$(FILE)"
	poetry run python tests/run_tests.py specific $(FILE) $(FUNCTION)

# 代碼品質檢查
format:
	@echo "🔧 格式化代碼..."
	poetry run black .
	poetry run isort .

lint:
	@echo "🔧 檢查代碼風格..."
	poetry run flake8 macos_video_auto_ocr_ass/ tests/

type-check:
	@echo "🔧 類型檢查..."
	poetry run mypy macos_video_auto_ocr_ass/

all-checks: format lint type-check
	@echo "✅ 所有代碼品質檢查完成"

# 清理
clean:
	@echo "🧹 清理暫存檔案..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "✅ 清理完成"

# 完整檢查
all-tests: all-checks test-coverage
	@echo "🎉 所有檢查完成！"

# 便捷的測試命令（使用 pytest 直接運行）
pytest:
	@echo "🧪 使用 pytest 直接運行測試..."
	poetry run pytest

pytest-verbose:
	@echo "🧪 使用 pytest 詳細運行測試..."
	poetry run pytest -v

pytest-watch:
	@echo "🧪 監控模式運行測試（需要安裝 pytest-watch）..."
	poetry run ptw

# 開發工具
shell:
	@echo "🐍 啟動 Poetry shell..."
	poetry shell

run-example:
	@echo "🚀 運行範例（請根據實際情況調整）..."
	@echo "可用的範例腳本："
	@echo "  - video_ocr_to_ass.py"
	@echo "  - translate_ass_llama.py"
	@echo "  - translate_ass_marianmt.py"
	@echo "  - merge_ass_subs.py"
	@echo "  - video_text_heatmap.py"

# 文檔相關
docs:
	@echo "📚 生成文檔（如果配置了）..."
	@echo "目前專案使用 README.md 作為主要文檔"

# 發布相關
build:
	@echo "📦 建構專案..."
	poetry build

publish:
	@echo "📤 發布到 PyPI（請確認版本號）..."
	poetry publish

# 依賴管理
update-deps:
	@echo "🔄 更新依賴..."
	poetry update

lock:
	@echo "🔒 更新 poetry.lock..."
	poetry lock

# 環境檢查
check-env:
	@echo "🔍 檢查環境..."
	@echo "Python 版本："
	@poetry run python --version
	@echo "Poetry 版本："
	@poetry --version
	@echo "已安裝的套件："
	@poetry show --tree

# 快速開發循環
dev: install-dev
	@echo "🚀 開發環境準備完成！"
	@echo "可用命令："
	@echo "  make test        - 運行測試"
	@echo "  make format      - 格式化代碼"
	@echo "  make all-checks  - 運行所有檢查" 