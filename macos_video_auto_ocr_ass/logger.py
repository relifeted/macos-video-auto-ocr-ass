"""
日誌系統模組

提供統一的日誌記錄功能
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日誌格式化器"""

    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 綠色
        "WARNING": "\033[33m",  # 黃色
        "ERROR": "\033[31m",  # 紅色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record: logging.LogRecord) -> str:
        # 取得原始 levelname
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS["RESET"])
        # 彩色 levelname
        record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
        # 彩色訊息
        record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "macos_video_auto_ocr_ass",
    level: str = "INFO",
    log_file: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """
    設置日誌記錄器

    Args:
        name: 日誌記錄器名稱
        level: 日誌級別
        log_file: 日誌檔案路徑（可選）
        colored: 是否使用彩色輸出

    Returns:
        配置好的日誌記錄器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除現有的處理器
    logger.handlers.clear()

    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    if colored:
        formatter: logging.Formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 檔案處理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "macos_video_auto_ocr_ass") -> logging.Logger:
    """
    獲取日誌記錄器

    Args:
        name: 日誌記錄器名稱

    Returns:
        日誌記錄器實例
    """
    return logging.getLogger(name)


# 預設日誌記錄器
logger = setup_logger()
