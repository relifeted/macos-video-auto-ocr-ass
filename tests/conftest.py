"""
pytest 配置檔案

包含共用的測試夾具和設定
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

from macos_video_auto_ocr_ass.config import AppConfig


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """創建臨時目錄"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_image() -> Image.Image:
    """創建測試用的樣本圖像"""
    # 創建一個簡單的測試圖像
    img = Image.new("RGB", (100, 50), color="white")
    return img


@pytest.fixture
def sample_ass_content() -> str:
    """創建測試用的 ASS 字幕內容"""
    return """[Script Info]
Title: Test Subtitle
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello World
Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,{\\pos(100,200)}Positioned text
"""


@pytest.fixture
def simple_ass_content() -> str:
    """創建簡單的 ASS 字幕內容（只有一個字幕事件）"""
    return """[Script Info]
Title: Test Subtitle
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello World
"""


@pytest.fixture
def sample_ass_file(sample_ass_content: str, temp_dir: str) -> str:
    """創建測試用的 ASS 檔案"""
    ass_file = os.path.join(temp_dir, "test.ass")
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(sample_ass_content)
    return ass_file


@pytest.fixture
def simple_ass_file(simple_ass_content: str, temp_dir: str) -> str:
    """創建簡單的 ASS 檔案（只有一個字幕事件）"""
    ass_file = os.path.join(temp_dir, "simple_test.ass")
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(simple_ass_content)
    return ass_file


@pytest.fixture
def app_config() -> AppConfig:
    """創建測試用的應用程式配置"""
    config = AppConfig()
    config.output_dir = "test_output"
    config.temp_dir = "test_temp"
    config.video.interval = 1.0
    config.video.downscale = 2
    config.translation.src_lang = "en"
    config.translation.tgt_lang = "Traditional Chinese"
    return config


@pytest.fixture
def mock_video_path() -> str:
    """模擬影片路徑（用於測試）"""
    return "/path/to/test/video.mp4"


@pytest.fixture
def mock_ocr_results() -> list:
    """模擬 OCR 結果"""
    return [
        ("Hello", None),  # 沒有邊界框的文字
        (
            "World",
            type(
                "BBox",
                (),
                {
                    "origin": type("Point", (), {"x": 0.1, "y": 0.2})(),
                    "size": type("Size", (), {"width": 0.3, "height": 0.1})(),
                },
            )(),
        ),  # 有邊界框的文字
    ]


class MockCGImage:
    """模擬 CGImage 物件"""

    def __init__(self, width: int = 100, height: int = 50):
        self.width = width
        self.height = height


class MockDataProvider:
    """模擬 CGDataProvider 物件"""

    def __init__(self, data: bytes):
        self.data = data


def create_mock_cgimage(width: int = 100, height: int = 50) -> MockCGImage:
    """創建模擬的 CGImage"""
    return MockCGImage(width, height)


def create_mock_data_provider() -> MockDataProvider:
    """創建模擬的 CGDataProvider"""
    # 創建模擬的圖像數據
    data = bytes([255] * 100 * 50 * 4)  # RGBA 格式
    return MockDataProvider(data)


# 測試標記
def pytest_configure(config):
    """配置測試標記"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# 跳過 macOS 特定測試的標記
def pytest_collection_modifyitems(config, items):
    """修改測試項目，標記 macOS 特定測試"""
    for item in items:
        if "macos" in item.nodeid.lower() or "vision" in item.nodeid.lower():
            item.add_marker(
                pytest.mark.skipif(
                    not os.name == "posix" or not os.uname().sysname == "Darwin",
                    reason="需要 macOS 系統",
                )
            )
