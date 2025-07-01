# macOS 影片自動 OCR 字幕工具 Makefile
# 精簡版

.PHONY: help install install-dev test format lint type-check check clean all

help:
	@echo "macOS 影片自動 OCR 字幕工具 - 可用命令："
	@echo "  install        - 安裝專案依賴"
	@echo "  install-dev    - 安裝開發依賴"
	@echo "  test           - 運行所有測試"
	@echo "  format         - 格式化代碼"
	@echo "  lint           - 檢查代碼風格"
	@echo "  type-check     - 類型檢查"
	@echo "  check          - 格式化+檢查+型別"
	@echo "  clean          - 清理暫存與編譯檔案"
	@echo "  video_ocr_to_json - 編譯 Objective-C CLI 工具"

install:
	poetry install --only main

install-dev:
	poetry install

test:
	poetry run python tests/run_tests.py all

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 macos_video_auto_ocr_ass/ tests/

type-check:
	poetry run mypy macos_video_auto_ocr_ass/

check: format lint type-check

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f video_ocr_to_json

CC = clang
CFLAGS = -framework Foundation -framework AVFoundation -framework CoreImage -framework Vision -framework AppKit -framework CoreMedia

video_ocr_to_json: video_ocr_to_json.m
	$(CC) $(CFLAGS) -o video_ocr_to_json video_ocr_to_json.m
	
all: video_ocr_to_json 

run-electron:
	cd electron_gui && npm start

run-api:
	poetry run python -m macos_video_auto_ocr_ass.ocr_api

