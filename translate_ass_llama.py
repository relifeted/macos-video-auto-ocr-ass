import argparse
import os
import re
from collections import defaultdict

import pysubs2
from langdetect import detect
from llama_cpp import Llama
from tqdm import tqdm

# 新增 opencc
try:
    from opencc import OpenCC

    opencc = OpenCC("s2t")
except ImportError:
    opencc = None


def download_model(model_repo, model_filename, local_dir):
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=model_repo, filename=model_filename, local_dir=local_dir
    )
    return model_path


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


def translate_with_llama(llm, text, src_lang, tgt_lang, retry=1):
    prompt = (
        f"Translate the following text from {src_lang} to {tgt_lang}.\n"
        f"Only output the translation in {tgt_lang}, do not use any other language or add explanation.\n"
        f"{text}\n"
        f"Translation:"
    )
    for _ in range(retry + 1):
        output = llm(prompt, max_tokens=512, stop=["\n"])
        result = output["choices"][0]["text"].strip()
        # 自動檢查語言
        if tgt_lang.lower() == "traditional chinese":
            try:
                if detect(result) == "zh-tw":
                    return result
            except Exception:
                pass
        else:
            return result
    return result  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ass", help="輸入 ASS 字幕檔")
    parser.add_argument("output_ass", help="輸出 ASS 字幕檔")
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="來源語言（預設 auto，會在 prompt 指定 auto-detect）",
    )
    parser.add_argument(
        "--tgt-lang",
        default="Traditional Chinese",
        help="目標語言（如 Traditional Chinese, Simplified Chinese, Japanese 等）",
    )
    parser.add_argument(
        "--model-repo",
        default="mradermacher/X-ALMA-13B-Group6-GGUF",
        help="Hugging Face 模型 repo",
    )
    parser.add_argument(
        "--model-filename", default="X-ALMA-13B-Group6.Q8_0.gguf", help="GGUF 檔名"
    )
    parser.add_argument("--model-dir", default="models", help="模型下載資料夾")
    parser.add_argument("--n-ctx", type=int, default=2048, help="llama-cpp n_ctx")
    parser.add_argument("--n-threads", type=int, default=4, help="llama-cpp n_threads")
    parser.add_argument("--show-text", action="store_true", help="顯示翻譯前後的文字")
    args = parser.parse_args()

    print("下載/載入模型中...")
    model_path = download_model(args.model_repo, args.model_filename, args.model_dir)

    llm = Llama(
        model_path=model_path, n_ctx=args.n_ctx, n_threads=args.n_threads, verbose=False
    )

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
        src_lang = args.src_lang if args.src_lang != "auto" else "auto-detect"
        translated = translate_with_llama(llm, merged_text, src_lang, args.tgt_lang)
        # 若目標語言為繁體中文，強制用 opencc 轉換
        if args.tgt_lang.lower() == "traditional chinese" and opencc is not None:
            translated = opencc.convert(translated)
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
