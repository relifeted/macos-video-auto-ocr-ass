"""
日誌系統測試

測試日誌記錄功能
"""

import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from macos_video_auto_ocr_ass.logger import ColoredFormatter, get_logger, setup_logger


class TestColoredFormatter:
    """測試彩色格式化器"""

    def test_colored_formatter_init(self):
        """測試彩色格式化器初始化"""
        formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        assert isinstance(formatter, ColoredFormatter)
        assert isinstance(formatter, logging.Formatter)

    def test_colored_formatter_format(self):
        """測試彩色格式化器格式化"""
        formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 創建模擬的記錄
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "test_logger" in formatted
        assert "Test message" in formatted
        assert "INFO" in formatted

    def test_colored_formatter_colors(self):
        """測試彩色格式化器顏色"""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")

        # 測試不同級別的顏色
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level_name in levels:
            record = logging.LogRecord(
                name="test_logger",
                level=getattr(logging, level_name),
                pathname="test.py",
                lineno=10,
                msg=f"Test {level_name} message",
                args=(),
                exc_info=None,
            )

            formatted = formatter.format(record)
            # 檢查是否包含顏色代碼
            assert "\033[" in formatted
            assert level_name in formatted


class TestSetupLogger:
    """測試日誌設置"""

    def test_setup_logger_default(self):
        """測試預設日誌設置"""
        logger = setup_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "macos_video_auto_ocr_ass"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

        # 檢查是否有控制台處理器
        console_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) >= 1

    def test_setup_logger_custom_name(self):
        """測試自訂名稱日誌設置"""
        logger = setup_logger(name="custom_logger")

        assert logger.name == "custom_logger"
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """測試自訂級別日誌設置"""
        logger = setup_logger(level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self, temp_dir):
        """測試帶檔案處理器的日誌設置"""
        log_file = os.path.join(temp_dir, "test.log")
        logger = setup_logger(log_file=log_file)

        # 檢查是否有檔案處理器
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1

        # 測試寫入日誌
        logger.info("Test message")

        # 檢查檔案是否被創建並包含日誌
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Test message" in content

    def test_setup_logger_no_color(self):
        """測試無彩色日誌設置"""
        logger = setup_logger(colored=False)

        # 檢查格式化器是否為普通格式化器
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                formatter = handler.formatter
                assert not isinstance(formatter, ColoredFormatter)
                break

    def test_setup_logger_clears_handlers(self):
        """測試日誌設置清除現有處理器"""
        # 先創建一個日誌記錄器
        logger1 = setup_logger(name="test_clear")
        initial_handler_count = len(logger1.handlers)

        # 再次設置同一個日誌記錄器
        logger2 = setup_logger(name="test_clear")

        # 應該清除舊的處理器並添加新的
        assert len(logger2.handlers) == initial_handler_count

    def test_setup_logger_invalid_level(self):
        """測試無效級別日誌設置"""
        # 使用 getattr 會拋出 AttributeError，而不是 ValueError
        with pytest.raises(AttributeError):
            setup_logger(level="INVALID_LEVEL")


class TestGetLogger:
    """測試獲取日誌記錄器"""

    def test_get_logger_default(self):
        """測試預設日誌記錄器獲取"""
        logger = get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "macos_video_auto_ocr_ass"

    def test_get_logger_custom_name(self):
        """測試自訂名稱日誌記錄器獲取"""
        logger = get_logger("custom_name")

        assert logger.name == "custom_name"

    def test_get_logger_same_instance(self):
        """測試相同名稱返回相同實例"""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")

        assert logger1 is logger2


class TestLoggerIntegration:
    """測試日誌整合功能"""

    def test_logger_info_message(self, temp_dir):
        """測試資訊日誌訊息"""
        log_file = os.path.join(temp_dir, "integration.log")
        logger = setup_logger(log_file=log_file)

        logger.info("Integration test message")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Integration test message" in content
            assert "INFO" in content

    def test_logger_error_message(self, temp_dir):
        """測試錯誤日誌訊息"""
        log_file = os.path.join(temp_dir, "error.log")
        logger = setup_logger(log_file=log_file)

        logger.error("Error test message")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Error test message" in content
            assert "ERROR" in content

    def test_logger_debug_message_filtered(self, temp_dir):
        """測試調試訊息被過濾"""
        log_file = os.path.join(temp_dir, "debug.log")
        logger = setup_logger(level="INFO", log_file=log_file)

        logger.debug("Debug message that should not appear")
        logger.info("Info message that should appear")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Debug message that should not appear" not in content
            assert "Info message that should appear" in content

    def test_logger_multiple_handlers(self, temp_dir):
        """測試多個處理器"""
        log_file = os.path.join(temp_dir, "multi.log")
        logger = setup_logger(log_file=log_file)

        # 應該有控制台和檔案處理器
        assert len(logger.handlers) >= 2

        # 測試訊息應該同時輸出到控制台和檔案
        logger.info("Multi-handler test")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Multi-handler test" in content


class TestLoggerEdgeCases:
    """測試日誌邊界情況"""

    def test_logger_empty_message(self, temp_dir):
        """測試空訊息"""
        log_file = os.path.join(temp_dir, "empty.log")
        logger = setup_logger(log_file=log_file)

        logger.info("")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 應該有時間戳和級別，但訊息為空
            assert "INFO" in content

    def test_logger_special_characters(self, temp_dir):
        """測試特殊字符"""
        log_file = os.path.join(temp_dir, "special.log")
        logger = setup_logger(log_file=log_file)

        special_message = "特殊字符: 中文、日本語、한국어"
        logger.info(special_message)

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert special_message in content

    def test_logger_long_message(self, temp_dir):
        """測試長訊息"""
        log_file = os.path.join(temp_dir, "long.log")
        logger = setup_logger(log_file=log_file)

        long_message = "A" * 1000
        logger.info(long_message)

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert long_message in content

    @patch("os.uname")
    def test_logger_macos_specific(self, mock_uname, temp_dir):
        """測試 macOS 特定功能"""
        # 模擬 macOS 系統
        mock_uname.return_value.sysname = "Darwin"

        log_file = os.path.join(temp_dir, "macos.log")
        logger = setup_logger(log_file=log_file)

        logger.info("macOS specific test")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "macOS specific test" in content
