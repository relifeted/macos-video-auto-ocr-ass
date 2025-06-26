"""
配置管理模組

統一管理應用程式的各種設定
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VideoConfig:
    """影片處理配置"""

    interval: float = 1.0
    downscale: int = 2
    quiet: bool = False
    scan_rect: Optional[Tuple[int, int, int, int]] = None
    x_offset: int = 0
    base_font_size: int = 24
    recognition_languages: Optional[List[str]] = None


@dataclass
class OCRConfig:
    """OCR 配置"""

    languages: Optional[List[str]] = None
    auto_detect: bool = True


@dataclass
class TranslationConfig:
    """翻譯配置"""

    src_lang: str = "auto"
    tgt_lang: str = "Traditional Chinese"
    translator_type: str = "llama"  # "llama" 或 "marianmt"
    show_text: bool = False

    # Llama 特定配置
    model_repo: str = "mradermacher/X-ALMA-13B-Group6-GGUF"
    model_filename: str = "X-ALMA-13B-Group6.Q8_0.gguf"
    model_dir: str = "models"
    n_ctx: int = 2048
    n_threads: int = 4

    # MarianMT 特定配置
    device: str = "mps"
    max_length: int = 1024


@dataclass
class MergeConfig:
    """字幕合併配置"""

    position_tolerance: int = 10
    time_gap_threshold: int = 500
    merge_events: bool = True


@dataclass
class HeatmapConfig:
    """熱區圖配置"""

    grid_interval: int = 300
    contrast_boost: float = 1.0


@dataclass
class AppConfig:
    """應用程式主配置"""

    video: VideoConfig = field(default_factory=VideoConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)

    # 全局配置
    output_dir: str = "output"
    temp_dir: str = "temp"
    log_level: str = "INFO"

    def __post_init__(self):
        """初始化後處理"""
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AppConfig":
        """從字典創建配置"""
        config = cls()

        # 更新影片配置
        if "video" in config_dict:
            for key, value in config_dict["video"].items():
                if hasattr(config.video, key):
                    setattr(config.video, key, value)

        # 更新 OCR 配置
        if "ocr" in config_dict:
            for key, value in config_dict["ocr"].items():
                if hasattr(config.ocr, key):
                    setattr(config.ocr, key, value)

        # 更新翻譯配置
        if "translation" in config_dict:
            for key, value in config_dict["translation"].items():
                if hasattr(config.translation, key):
                    setattr(config.translation, key, value)

        # 更新合併配置
        if "merge" in config_dict:
            for key, value in config_dict["merge"].items():
                if hasattr(config.merge, key):
                    setattr(config.merge, key, value)

        # 更新熱區圖配置
        if "heatmap" in config_dict:
            for key, value in config_dict["heatmap"].items():
                if hasattr(config.heatmap, key):
                    setattr(config.heatmap, key, value)

        # 更新全局配置
        for key, value in config_dict.items():
            if key not in ["video", "ocr", "translation", "merge", "heatmap"]:
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            "video": {
                "interval": self.video.interval,
                "downscale": self.video.downscale,
                "quiet": self.video.quiet,
                "scan_rect": self.video.scan_rect,
                "x_offset": self.video.x_offset,
                "base_font_size": self.video.base_font_size,
                "recognition_languages": self.video.recognition_languages,
            },
            "ocr": {
                "languages": self.ocr.languages,
                "auto_detect": self.ocr.auto_detect,
            },
            "translation": {
                "src_lang": self.translation.src_lang,
                "tgt_lang": self.translation.tgt_lang,
                "translator_type": self.translation.translator_type,
                "show_text": self.translation.show_text,
                "model_repo": self.translation.model_repo,
                "model_filename": self.translation.model_filename,
                "model_dir": self.translation.model_dir,
                "n_ctx": self.translation.n_ctx,
                "n_threads": self.translation.n_threads,
                "device": self.translation.device,
                "max_length": self.translation.max_length,
            },
            "merge": {
                "position_tolerance": self.merge.position_tolerance,
                "time_gap_threshold": self.merge.time_gap_threshold,
                "merge_events": self.merge.merge_events,
            },
            "heatmap": {
                "grid_interval": self.heatmap.grid_interval,
                "contrast_boost": self.heatmap.contrast_boost,
            },
            "output_dir": self.output_dir,
            "temp_dir": self.temp_dir,
            "log_level": self.log_level,
        }


def load_config(config_file: str) -> AppConfig:
    """從檔案載入配置"""
    import json

    if not os.path.exists(config_file):
        return AppConfig()

    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return AppConfig.from_dict(config_dict)


def save_config(config: AppConfig, config_file: str) -> None:
    """儲存配置到檔案"""
    import json

    config_dict = config.to_dict()

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


# 預設配置
DEFAULT_CONFIG = AppConfig()
