import argparse
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pysubs2
from huggingface_hub.utils import RepositoryNotFoundError
from langdetect import detect
from tqdm import tqdm
from transformers import pipeline

from macos_video_auto_ocr_ass.constants import (
    LOGGER_NAME,
    MARIANMT_DEFAULT_DEVICE,
    MARIANMT_DEFAULT_SRC_LANG,
    MARIANMT_DEFAULT_TGT_LANG,
    MARIANMT_MAX_LENGTH,
    MARIANMT_SPLIT_THRESHOLD,
)
from macos_video_auto_ocr_ass.logger import get_logger

# 新增 opencc
try:
    from opencc import OpenCC

    opencc = OpenCC("s2t")
except ImportError:
    opencc = None

logger = get_logger(LOGGER_NAME)


def extract_text_and_tags(text: str) -> Tuple[str, List[str]]:
    tags = re.findall(r"{.*?}", text)
    text_only = re.sub(r"{.*?}", "", text)
    return text_only, tags


def restore_tags(translated: str, tags: List[str]) -> str:
    return "".join(tags) + translated


def group_by_time(subs: Any) -> Dict[Tuple[int, int], List[Any]]:
    groups = defaultdict(list)
    for line in subs:
        key = (line.start, line.end)
        groups[key].append(line)
    return groups


def get_translator(src: str, tgt: str, device: str) -> Optional[Any]:
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        return pipeline("translation", model=model_name, device=device)
    except (OSError, RepositoryNotFoundError):
        return None


def load_translators(
    src_lang: str, tgt_lang: str, device: str
) -> Tuple[Dict[str, Any], str, bool]:
    """預先載入所有需要的翻譯模型"""
    translators = {}
    tgt_lang_for_model = tgt_lang
    if tgt_lang.lower() in ["zht", "zh", "zhs"]:
        tgt_lang_for_model = "zh"
    direct_translator = get_translator(src_lang, tgt_lang_for_model, device)
    if direct_translator:
        translators["direct"] = direct_translator
        logger.info(f"載入直接翻譯模型: opus-mt-{src_lang}-{tgt_lang_for_model}")
        return translators, tgt_lang_for_model, False
    if src_lang != "en" and tgt_lang_for_model != "en":
        src2en = get_translator(src_lang, "en", device)
        en2tgt = get_translator("en", tgt_lang_for_model, device)
        if src2en and en2tgt:
            translators["src2en"] = src2en
            translators["en2tgt"] = en2tgt
            logger.info(
                f"載入中轉翻譯模型: opus-mt-{src_lang}-en + opus-mt-en-{tgt_lang_for_model}"
            )
            return translators, tgt_lang_for_model, True
    raise RuntimeError(f"找不到可用的 {src_lang}->{tgt_lang} 翻譯模型，也無法中轉。")


def translate_with_loaded_models(
    text: str,
    translators: Dict[str, Any],
    need_opencc: bool,
    max_length: int = MARIANMT_MAX_LENGTH,
) -> str:
    """使用預先載入的模型進行翻譯，max_length為512，僅當長度超過450才斷句"""
    fixed_max_length = MARIANMT_MAX_LENGTH
    split_threshold = MARIANMT_SPLIT_THRESHOLD  # 只有超過 450 才斷句

    def split_text_into_chunks(
        text: str, max_chunk_length: int, threshold: int
    ) -> List[str]:
        if len(text) <= threshold:
            return [text]
        sentences = re.split(r"[.!?。！？\n]", text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) <= max_chunk_length:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                if len(current_chunk + " " + sentence) <= max_chunk_length:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    text_chunks = split_text_into_chunks(text, fixed_max_length, split_threshold)
    translated_parts = []
    for chunk in text_chunks:
        if not chunk.strip():
            continue
        try:
            if "direct" in translators:
                translated_part = translators["direct"](
                    chunk, max_length=fixed_max_length
                )[0]["translation_text"].strip()
            elif "src2en" in translators and "en2tgt" in translators:
                en_text = translators["src2en"](chunk, max_length=fixed_max_length)[0][
                    "translation_text"
                ].strip()
                translated_part = translators["en2tgt"](
                    en_text, max_length=fixed_max_length
                )[0]["translation_text"].strip()
            else:
                raise RuntimeError("沒有可用的翻譯模型")
            translated_parts.append(translated_part)
        except Exception as e:
            logger.warning(f"翻譯片段失敗，使用原文: {e}")
            translated_parts.append(chunk)
    translated = " ".join(translated_parts)
    if need_opencc and opencc is not None:
        translated = opencc.convert(translated)
    return translated


def detect_lang(text: str) -> Optional[str]:
    try:
        return detect(text)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ass", help="輸入 ASS 字幕檔")
    parser.add_argument("output_ass", help="輸出 ASS 字幕檔")
    parser.add_argument(
        "--src-lang",
        default=MARIANMT_DEFAULT_SRC_LANG,
        help="來源語言（ISO 代碼，如 en, ja, zh, ko）",
    )
    parser.add_argument(
        "--tgt-lang",
        default=MARIANMT_DEFAULT_TGT_LANG,
        help="目標語言（zht=繁體, zhs=簡體, zh=繁體，全部都用 zh 模型）",
    )
    parser.add_argument(
        "--model", default=None, help="Hugging Face MarianMT 模型名稱，預設自動推斷"
    )
    parser.add_argument(
        "--device",
        default=MARIANMT_DEFAULT_DEVICE,
        help="運算裝置（如 cpu, mps, cuda）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MARIANMT_MAX_LENGTH,
        help="翻譯最大長度（已固定為512以避免模型警告，此參數僅為兼容性保留）",
    )
    parser.add_argument("--show-text", action="store_true", help="顯示翻譯前後的文字")
    args = parser.parse_args()
    need_opencc = False
    if args.tgt_lang.lower() in ["zht", "zh"]:
        need_opencc = True
    elif args.tgt_lang.lower() == "zhs":
        need_opencc = False
    src_is_zh = args.src_lang.lower() in ["zh", "zhs", "zht"]
    tgt_is_zh = args.tgt_lang.lower() in ["zh", "zhs", "zht"]
    zh_only_mode = src_is_zh and tgt_is_zh
    if not zh_only_mode:
        logger.info("載入翻譯模型中...")
        try:
            translators, tgt_lang_for_model, use_pivot = load_translators(
                args.src_lang, args.tgt_lang, args.device
            )
        except RuntimeError as e:
            logger.error(f"{e}")
            return
    else:
        logger.info("zh 相關語言模式，跳過翻譯模型載入")
        translators = None
    subs = pysubs2.load(args.input_ass)
    new_lines = []
    for line in tqdm(subs, desc="Translating"):
        text, tags = extract_text_and_tags(line.text)
        text = text.strip()
        detected_lang = detect_lang(text)
        src_is_zh = args.src_lang.lower() in ["zh", "zhs", "zht"]
        tgt_is_zh = args.tgt_lang.lower() in ["zh", "zhs", "zht"]
        if src_is_zh and tgt_is_zh:
            if args.tgt_lang.lower() in ["zht", "zh"] and opencc is not None:
                converted = opencc.convert(text)
            else:
                converted = text
            new_line = line.copy()
            new_line.text = restore_tags(converted, tags)
            new_lines.append(new_line)
            continue
        if detected_lang and detected_lang != args.src_lang:
            new_line = line.copy()
            new_line.text = restore_tags(text, tags)
            new_lines.append(new_line)
            continue
        if len(text) > MARIANMT_SPLIT_THRESHOLD:
            logger.info(
                f"文本長度 {len(text)} 超過{MARIANMT_SPLIT_THRESHOLD}，將進行自動斷句處理"
            )
        try:
            translated = translate_with_loaded_models(text, translators, need_opencc)
        except RuntimeError as e:
            logger.error(f"翻譯失敗: {e}")
            translated = text  # fallback: 不翻譯
        except Exception as e:
            logger.error(f"翻譯過程中發生未預期錯誤: {e}")
            translated = text  # fallback: 不翻譯
        if args.show_text:
            logger.info("--- 原文 ---")
            logger.info(text)
            logger.info("--- 譯文 ---")
            logger.info(translated)
        new_line = line.copy()
        new_line.text = restore_tags(translated, tags)
        new_lines.append(new_line)
    new_lines.sort(key=lambda l: (l.start, l.end))
    subs.events = new_lines
    subs.save(args.output_ass)


if __name__ == "__main__":
    main()
