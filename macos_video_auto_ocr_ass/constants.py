"""
專案常數集中管理
"""

# Logger 名稱
LOGGER_NAME = "macos_video_auto_ocr_ass"

# 預設模型 repo 與檔名
DEFAULT_LLAMA_MODEL_REPO = "mradermacher/X-ALMA-13B-Group6-GGUF"
DEFAULT_LLAMA_MODEL_FILENAME = "X-ALMA-13B-Group6.Q8_0.gguf"
DEFAULT_MODEL_DIR = "models"

# 預設 context
DEFAULT_N_CTX = 2048
DEFAULT_N_THREADS = 4

# MarianMT pipeline 名稱
MARIANMT_PIPELINE = "translation"

# ASS 標籤正則
ASS_TAG_REGEX = r"{.*?}"
ASS_POS_REGEX = r"\\pos\(([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\)"

# 影片處理 magic number
DEFAULT_INTERVAL = 1.0
DEFAULT_DOWNSCALE = 2
DEFAULT_FONT_SIZE = 24

# 其他
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_TEMP_DIR = "temp"
