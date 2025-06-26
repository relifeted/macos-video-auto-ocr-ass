import argparse
import os
import re
from collections import defaultdict

import pysubs2
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines import Pipeline

# 新增 opencc
try:
    from opencc import OpenCC

    opencc = OpenCC("s2t")
except ImportError:
    opencc = None


def extract_text_and_tags(text):
    tags = re.findall(r"{.*?}", text)
    text_only = re.sub(r"{.*?}", "", text)
    return text_only, tags


def restore_tags(translated, tags):
    return "".join(tags) + translated


def group_by_time(subs):
    groups = defaultdict(list)
    for line in subs:
        key = (line.start, line.end)
        groups[key].append(line)
    return groups


def get_translator(src, tgt, device):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        return pipeline("translation", model=model_name, device=device)
    except (OSError, RepositoryNotFoundError):
        return None


def load_translators(src_lang, tgt_lang, device):
    """預先載入所有需要的翻譯模型"""
    translators = {}

    # 處理目標語言
    tgt_lang_for_model = tgt_lang
    if tgt_lang.lower() in ["zht", "zh", "zhs"]:
        tgt_lang_for_model = "zh"

    # 嘗試直接 src-tgt
    direct_translator = get_translator(src_lang, tgt_lang_for_model, device)
    if direct_translator:
        translators["direct"] = direct_translator
        print(f"載入直接翻譯模型: opus-mt-{src_lang}-{tgt_lang_for_model}")
        return translators, tgt_lang_for_model, False

    # 若找不到，嘗試 src-en + en-tgt
    if src_lang != "en" and tgt_lang_for_model != "en":
        src2en = get_translator(src_lang, "en", device)
        en2tgt = get_translator("en", tgt_lang_for_model, device)
        if src2en and en2tgt:
            translators["src2en"] = src2en
            translators["en2tgt"] = en2tgt
            print(
                f"載入中轉翻譯模型: opus-mt-{src_lang}-en + opus-mt-en-{tgt_lang_for_model}"
            )
            return translators, tgt_lang_for_model, True

    raise RuntimeError(f"找不到可用的 {src_lang}->{tgt_lang} 翻譯模型，也無法中轉。")


def translate_with_loaded_models(text, translators, need_opencc, max_length=512):
    """使用預先載入的模型進行翻譯，max_length為512，僅當長度超過450才斷句"""
    fixed_max_length = 512
    split_threshold = 450  # 只有超過 450 才斷句

    # 預先斷句處理
    def split_text_into_chunks(text, max_chunk_length, threshold):
        """將文本分割成不超過指定長度的片段，僅當超過 threshold 才分割"""
        if len(text) <= threshold:
            return [text]

        # 按句子分割
        sentences = re.split(r"[.!?。！？\n]", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果單個句子就超過限制，按詞分割
            if len(sentence) > max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 按詞分割長句子
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
                # 檢查加上這個句子是否會超過限制
                if len(current_chunk + " " + sentence) <= max_chunk_length:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        # 添加最後一個片段
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # 只有超過 split_threshold 才分割
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
            print(f"[WARNING] 翻譯片段失敗，使用原文: {e}")
            translated_parts.append(chunk)

    # 合併翻譯結果
    translated = " ".join(translated_parts)

    if need_opencc and opencc is not None:
        translated = opencc.convert(translated)
    return translated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ass", help="輸入 ASS 字幕檔")
    parser.add_argument("output_ass", help="輸出 ASS 字幕檔")
    parser.add_argument(
        "--src-lang", default="en", help="來源語言（ISO 代碼，如 en, ja, zh, ko）"
    )
    parser.add_argument(
        "--tgt-lang",
        default="zh",
        help="目標語言（zht=繁體, zhs=簡體, zh=繁體，全部都用 zh 模型）",
    )
    parser.add_argument(
        "--model", default=None, help="Hugging Face MarianMT 模型名稱，預設自動推斷"
    )
    parser.add_argument("--device", default="mps", help="運算裝置（如 cpu, mps, cuda）")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="翻譯最大長度（已固定為512以避免模型警告，此參數僅為兼容性保留）",
    )
    parser.add_argument("--show-text", action="store_true", help="顯示翻譯前後的文字")
    args = parser.parse_args()

    # 處理目標語言與 opencc
    need_opencc = False
    if args.tgt_lang.lower() in ["zht", "zh"]:
        need_opencc = True
    elif args.tgt_lang.lower() == "zhs":
        need_opencc = False

    # 預先載入翻譯模型
    print("載入翻譯模型中...")
    try:
        translators, tgt_lang_for_model, use_pivot = load_translators(
            args.src_lang, args.tgt_lang, args.device
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    subs = pysubs2.load(args.input_ass)
    groups = group_by_time(subs)
    new_lines = []

    for (start, end), lines in tqdm(groups.items(), desc="Translating"):
        texts, tags_list = [], []
        for line in lines:
            text, tags = extract_text_and_tags(line.text)
            texts.append(text.strip())
            tags_list.append(tags)
        merged_text = " ".join(texts)

        # 檢查文本長度並記錄
        if len(merged_text) > 450:
            print(f"[INFO] 文本長度 {len(merged_text)} 超過450，將進行自動斷句處理")

        try:
            translated = translate_with_loaded_models(
                merged_text, translators, need_opencc
            )
        except RuntimeError as e:
            print(f"[ERROR] 翻譯失敗: {e}")
            translated = merged_text  # fallback: 不翻譯
        except Exception as e:
            print(f"[ERROR] 翻譯過程中發生未預期錯誤: {e}")
            translated = merged_text  # fallback: 不翻譯

        if args.show_text:
            print("--- 原文 ---")
            print(merged_text)
            print("--- 譯文 ---")
            print(translated)
            print()
        translated_lines = translated.split(" ")
        for i, line in enumerate(lines):
            if i == 0:
                line.text = restore_tags(" ".join(translated_lines), tags_list[i])
            else:
                line.text = restore_tags("", tags_list[i])
            new_lines.append(line)

    new_lines.sort(key=lambda l: (l.start, l.end))
    subs.events = new_lines
    subs.save(args.output_ass)


if __name__ == "__main__":
    main()
