"""
配置模組測試

測試配置管理功能
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from macos_video_auto_ocr_ass.config import (
    DEFAULT_CONFIG,
    AppConfig,
    HeatmapConfig,
    MergeConfig,
    OCRConfig,
    TranslationConfig,
    VideoConfig,
    load_config,
    save_config,
)


class TestVideoConfig:
    """測試影片配置"""

    def test_video_config_defaults(self):
        """測試影片配置預設值"""
        config = VideoConfig()
        assert config.interval == 1.0
        assert config.downscale == 2
        assert config.quiet is False
        assert config.scan_rect is None
        assert config.x_offset == 0
        assert config.base_font_size == 24
        assert config.recognition_languages is None

    def test_video_config_custom(self):
        """測試自訂影片配置"""
        config = VideoConfig(
            interval=0.5,
            downscale=4,
            quiet=True,
            scan_rect=(10, 20, 100, 50),
            x_offset=5,
            base_font_size=32,
            recognition_languages=["en", "zh"],
        )
        assert config.interval == 0.5
        assert config.downscale == 4
        assert config.quiet is True
        assert config.scan_rect == (10, 20, 100, 50)
        assert config.x_offset == 5
        assert config.base_font_size == 32
        assert config.recognition_languages == ["en", "zh"]


class TestOCRConfig:
    """測試 OCR 配置"""

    def test_ocr_config_defaults(self):
        """測試 OCR 配置預設值"""
        config = OCRConfig()
        assert config.languages is None
        assert config.auto_detect is True

    def test_ocr_config_custom(self):
        """測試自訂 OCR 配置"""
        config = OCRConfig(languages=["en", "zh"], auto_detect=False)
        assert config.languages == ["en", "zh"]
        assert config.auto_detect is False


class TestTranslationConfig:
    """測試翻譯配置"""

    def test_translation_config_defaults(self):
        """測試翻譯配置預設值"""
        config = TranslationConfig()
        assert config.src_lang == "auto"
        assert config.tgt_lang == "Traditional Chinese"
        assert config.translator_type == "llama"
        assert config.show_text is False
        assert config.model_repo == "mradermacher/X-ALMA-13B-Group6-GGUF"
        assert config.model_filename == "X-ALMA-13B-Group6.Q8_0.gguf"
        assert config.model_dir == "models"
        assert config.n_ctx == 2048
        assert config.n_threads == 4
        assert config.device == "mps"
        assert config.max_length == 1024

    def test_translation_config_custom(self):
        """測試自訂翻譯配置"""
        config = TranslationConfig(
            src_lang="en",
            tgt_lang="Japanese",
            translator_type="marianmt",
            show_text=True,
            model_repo="custom/repo",
            model_filename="custom.gguf",
            model_dir="custom_models",
            n_ctx=4096,
            n_threads=8,
            device="cpu",
            max_length=2048,
        )
        assert config.src_lang == "en"
        assert config.tgt_lang == "Japanese"
        assert config.translator_type == "marianmt"
        assert config.show_text is True
        assert config.model_repo == "custom/repo"
        assert config.model_filename == "custom.gguf"
        assert config.model_dir == "custom_models"
        assert config.n_ctx == 4096
        assert config.n_threads == 8
        assert config.device == "cpu"
        assert config.max_length == 2048


class TestMergeConfig:
    """測試合併配置"""

    def test_merge_config_defaults(self):
        """測試合併配置預設值"""
        config = MergeConfig()
        assert config.position_tolerance == 10
        assert config.time_gap_threshold == 500
        assert config.merge_events is True

    def test_merge_config_custom(self):
        """測試自訂合併配置"""
        config = MergeConfig(
            position_tolerance=20, time_gap_threshold=1000, merge_events=False
        )
        assert config.position_tolerance == 20
        assert config.time_gap_threshold == 1000
        assert config.merge_events is False


class TestHeatmapConfig:
    """測試熱區圖配置"""

    def test_heatmap_config_defaults(self):
        """測試熱區圖配置預設值"""
        config = HeatmapConfig()
        assert config.grid_interval == 50
        assert config.contrast_boost == 1.5

    def test_heatmap_config_custom(self):
        """測試自訂熱區圖配置"""
        config = HeatmapConfig(grid_interval=500, contrast_boost=2.0)
        assert config.grid_interval == 500
        assert config.contrast_boost == 2.0


class TestAppConfig:
    """測試應用程式配置"""

    def test_app_config_defaults(self):
        """測試應用程式配置預設值"""
        config = AppConfig()
        assert config.output_dir == "output"
        assert config.temp_dir == "temp"
        assert config.log_level == "INFO"
        assert isinstance(config.video, VideoConfig)
        assert isinstance(config.ocr, OCRConfig)
        assert isinstance(config.translation, TranslationConfig)
        assert isinstance(config.merge, MergeConfig)
        assert isinstance(config.heatmap, HeatmapConfig)

    def test_app_config_custom(self):
        """測試自訂應用程式配置"""
        config = AppConfig()
        config.output_dir = "custom_output"
        config.temp_dir = "custom_temp"
        config.log_level = "DEBUG"
        config.video.interval = 0.5
        config.translation.tgt_lang = "Japanese"

        assert config.output_dir == "custom_output"
        assert config.temp_dir == "custom_temp"
        assert config.log_level == "DEBUG"
        assert config.video.interval == 0.5
        assert config.translation.tgt_lang == "Japanese"

    def test_app_config_from_dict(self):
        """測試從字典創建配置"""
        config_dict = {
            "video": {"interval": 0.5, "downscale": 4, "quiet": True},
            "translation": {
                "src_lang": "en",
                "tgt_lang": "Japanese",
                "translator_type": "marianmt",
            },
            "output_dir": "custom_output",
            "log_level": "DEBUG",
        }

        config = AppConfig.from_dict(config_dict)
        assert config.video.interval == 0.5
        assert config.video.downscale == 4
        assert config.video.quiet is True
        assert config.translation.src_lang == "en"
        assert config.translation.tgt_lang == "Japanese"
        assert config.translation.translator_type == "marianmt"
        assert config.output_dir == "custom_output"
        assert config.log_level == "DEBUG"

    def test_app_config_to_dict(self):
        """測試配置轉換為字典"""
        config = AppConfig()
        config.video.interval = 0.5
        config.translation.tgt_lang = "Japanese"

        config_dict = config.to_dict()

        assert config_dict["video"]["interval"] == 0.5
        assert config_dict["translation"]["tgt_lang"] == "Japanese"
        assert config_dict["output_dir"] == "output"
        assert config_dict["log_level"] == "INFO"

    def test_app_config_creates_directories(self, temp_dir):
        """測試配置創建目錄"""
        config = AppConfig()
        config.output_dir = os.path.join(temp_dir, "output")
        config.temp_dir = os.path.join(temp_dir, "temp")

        # 重新初始化以觸發 __post_init__
        config.__post_init__()

        assert os.path.exists(config.output_dir)
        assert os.path.exists(config.temp_dir)


class TestConfigFactories:
    """測試配置工廠函數"""

    def test_create_default_video_config(self):
        """測試創建預設影片配置"""
        from macos_video_auto_ocr_ass.config import _create_default_video_config

        config = _create_default_video_config()
        assert isinstance(config, VideoConfig)
        assert config.interval == 1.0
        assert config.downscale == 2
        assert config.quiet is False
        assert config.scan_rect is None
        assert config.x_offset == 0
        assert config.base_font_size == 24
        assert config.recognition_languages is None

    def test_create_default_ocr_config(self):
        """測試創建預設 OCR 配置"""
        from macos_video_auto_ocr_ass.config import _create_default_ocr_config

        config = _create_default_ocr_config()
        assert isinstance(config, OCRConfig)
        assert config.languages is None
        assert config.auto_detect is True

    def test_create_default_translation_config(self):
        """測試創建預設翻譯配置"""
        from macos_video_auto_ocr_ass.config import _create_default_translation_config

        config = _create_default_translation_config()
        assert isinstance(config, TranslationConfig)
        assert config.src_lang == "auto"
        assert config.tgt_lang == "Traditional Chinese"
        assert config.translator_type == "llama"
        assert config.show_text is False
        assert config.model_repo == "mradermacher/X-ALMA-13B-Group6-GGUF"
        assert config.model_filename == "X-ALMA-13B-Group6.Q8_0.gguf"
        assert config.model_dir == "models"
        assert config.n_ctx == 2048
        assert config.n_threads == 4
        assert config.device == "mps"
        assert config.max_length == 1024

    def test_create_default_merge_config(self):
        """測試創建預設合併配置"""
        from macos_video_auto_ocr_ass.config import _create_default_merge_config

        config = _create_default_merge_config()
        assert isinstance(config, MergeConfig)
        assert config.position_tolerance == 10
        assert config.time_gap_threshold == 500
        assert config.merge_events is True

    def test_create_default_heatmap_config(self):
        """測試創建預設熱區圖配置"""
        from macos_video_auto_ocr_ass.config import _create_default_heatmap_config

        config = _create_default_heatmap_config()
        assert isinstance(config, HeatmapConfig)
        assert config.grid_interval == 50
        assert config.contrast_boost == 1.5

    def test_create_config_factory(self):
        """測試配置工廠函數"""
        from macos_video_auto_ocr_ass.config import create_config_factory

        config = create_config_factory()
        assert isinstance(config, AppConfig)
        assert isinstance(config.video, VideoConfig)
        assert isinstance(config.ocr, OCRConfig)
        assert isinstance(config.translation, TranslationConfig)
        assert isinstance(config.merge, MergeConfig)
        assert isinstance(config.heatmap, HeatmapConfig)
        assert config.output_dir == "output"
        assert config.temp_dir == "temp"
        assert config.log_level == "INFO"


class TestConfigFileOperations:
    """測試配置檔案操作"""

    def test_save_config_creates_directory(self, temp_dir):
        """測試儲存配置時創建目錄"""
        from macos_video_auto_ocr_ass.config import save_config

        config = AppConfig()
        config.output_dir = "custom_output"

        # 測試儲存到子目錄
        config_file = os.path.join(temp_dir, "subdir", "test_config.json")
        save_config(config, config_file)

        assert os.path.exists(config_file)
        assert os.path.exists(os.path.dirname(config_file))

    def test_save_config_json_format(self, temp_dir):
        """測試儲存配置的 JSON 格式"""
        from macos_video_auto_ocr_ass.config import load_config, save_config

        config = AppConfig()
        config.video.interval = 0.5
        config.translation.tgt_lang = "Japanese"
        config.output_dir = "custom_output"

        config_file = os.path.join(temp_dir, "test_config.json")
        save_config(config, config_file)

        # 檢查 JSON 格式
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        assert "video" in config_data
        assert "translation" in config_data
        assert "output_dir" in config_data
        assert config_data["video"]["interval"] == 0.5
        assert config_data["translation"]["tgt_lang"] == "Japanese"
        assert config_data["output_dir"] == "custom_output"

        # 測試載入
        loaded_config = load_config(config_file)
        assert loaded_config.video.interval == 0.5
        assert loaded_config.translation.tgt_lang == "Japanese"
        assert loaded_config.output_dir == "custom_output"

    def test_default_config_consistency(self):
        """測試預設配置的一致性"""
        from macos_video_auto_ocr_ass.config import (
            DEFAULT_CONFIG,
            create_config_factory,
        )

        # 測試預設配置與工廠函數的一致性
        factory_config = create_config_factory()
        assert factory_config.video.interval == DEFAULT_CONFIG.video.interval
        assert (
            factory_config.translation.tgt_lang == DEFAULT_CONFIG.translation.tgt_lang
        )
        assert factory_config.output_dir == DEFAULT_CONFIG.output_dir

    def test_default_config_usage(self):
        """測試 DEFAULT_CONFIG 的使用"""
        from macos_video_auto_ocr_ass.config import DEFAULT_CONFIG

        # 測試 DEFAULT_CONFIG 可以被正常使用
        assert DEFAULT_CONFIG is not None
        assert hasattr(DEFAULT_CONFIG, "video")
        assert hasattr(DEFAULT_CONFIG, "translation")
        assert hasattr(DEFAULT_CONFIG, "output_dir")

        # 測試可以修改 DEFAULT_CONFIG 的屬性
        original_interval = DEFAULT_CONFIG.video.interval
        DEFAULT_CONFIG.video.interval = 2.0
        assert DEFAULT_CONFIG.video.interval == 2.0

        # 恢復原值
        DEFAULT_CONFIG.video.interval = original_interval


class TestDefaultConfig:
    """測試預設配置"""

    def test_default_config(self):
        """測試預設配置"""
        assert isinstance(DEFAULT_CONFIG, AppConfig)
        assert DEFAULT_CONFIG.video.interval == 1.0
        assert DEFAULT_CONFIG.translation.tgt_lang == "Traditional Chinese"
        assert DEFAULT_CONFIG.output_dir == "output"
