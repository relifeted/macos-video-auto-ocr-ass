import sys
from typing import List, Optional

import objc
from Vision import VNRecognizeTextRequest

from macos_video_auto_ocr_ass.constants import LOGGER_NAME
from macos_video_auto_ocr_ass.logger import get_logger

logger = get_logger(LOGGER_NAME)


def main() -> None:
    # 建立一個 instance
    request = VNRecognizeTextRequest.alloc().init()
    # 呼叫 instance method，傳入 None 當 error 參數
    result = request.supportedRecognitionLanguagesAndReturnError_(None)
    # result 可能是 tuple (languages, error)
    if isinstance(result, tuple):
        languages, error = result
    else:
        languages = result
        error = None
    if error is not None:
        logger.error(f"取得語言時發生錯誤: {error}")
        return
    logger.info("支援的語言：")
    for lang in languages:
        logger.info(lang)


if __name__ == "__main__":
    main()
