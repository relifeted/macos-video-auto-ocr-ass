"""
翻譯器模組

提供統一的翻譯介面，支援多種翻譯引擎
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import pysubs2
from langdetect import detect

# 嘗試導入 opencc
try:
    from opencc import OpenCC

    opencc = OpenCC("s2t")
except ImportError:
    opencc = None


class BaseTranslator(ABC):
    """翻譯器抽象基類"""

    def __init__(self, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._model = None

    @abstractmethod
    def _load_model(self) -> None:
        """載入翻譯模型"""
        pass

    @abstractmethod
    def _translate_text(self, text: str) -> str:
        """翻譯單個文字"""
        pass

    def translate(self, text: str, retry: int = 1) -> str:
        """
        翻譯文字

        Args:
            text: 要翻譯的文字
            retry: 重試次數

        Returns:
            翻譯後的文字
        """
        if self._model is None:
            self._load_model()

        for _ in range(retry + 1):
            try:
                result = self._translate_text(text)

                # 自動檢查語言（如果目標語言是繁體中文）
                if self.tgt_lang.lower() == "traditional chinese":
                    try:
                        if detect(result) == "zh-tw":
                            return result
                    except Exception:
                        pass
                else:
                    return result
            except Exception as e:
                print(f"[WARNING] Translation failed: {e}")
                if _ == retry:  # 最後一次重試
                    return text  # 返回原文

        return text  # fallback

    def translate_ass_file(
        self, input_ass: str, output_ass: str, show_text: bool = False
    ) -> None:
        """
        翻譯 ASS 字幕檔案

        Args:
            input_ass: 輸入 ASS 檔案路徑
            output_ass: 輸出 ASS 檔案路徑
            show_text: 是否顯示翻譯前後文字
        """
        from .ass_utils import extract_text_and_tags, group_by_time, restore_tags

        subs = pysubs2.load(input_ass)
        groups = group_by_time(subs)
        new_lines = []

        for (start, end), lines in groups.items():
            texts, tags_list = [], []
            for line in lines:
                text, tags = extract_text_and_tags(line.text)
                texts.append(text.strip())
                tags_list.append(tags)

            merged_text = " ".join(texts)
            translated = self.translate(merged_text)

            # 若目標語言為繁體中文，強制用 opencc 轉換
            if self.tgt_lang.lower() == "traditional chinese" and opencc is not None:
                translated = opencc.convert(translated)

            if show_text:
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
        subs.save(output_ass)


class LlamaTranslator(BaseTranslator):
    """Llama 翻譯器"""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_repo: str = "mradermacher/X-ALMA-13B-Group6-GGUF",
        model_filename: str = "X-ALMA-13B-Group6.Q8_0.gguf",
        model_dir: str = "models",
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        super().__init__(src_lang, tgt_lang)
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.model_dir = model_dir
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    def _download_model(self) -> str:
        """下載模型"""
        from huggingface_hub import hf_hub_download

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_filename,
            local_dir=self.model_dir,
        )
        return model_path

    def _load_model(self) -> None:
        """載入 Llama 模型"""
        from llama_cpp import Llama

        print("下載/載入 Llama 模型中...")
        model_path = self._download_model()
        self._model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

    def _translate_text(self, text: str) -> str:
        """使用 Llama 翻譯文字"""
        src_lang = self.src_lang if self.src_lang != "auto" else "auto-detect"
        prompt = (
            f"Translate the following text from {src_lang} to {self.tgt_lang}.\n"
            f"Only output the translation in {self.tgt_lang}, do not use any other language or add explanation.\n"
            f"{text}\n"
            f"Translation:"
        )

        output = self._model(prompt, max_tokens=512, stop=["\n"])
        result = output["choices"][0]["text"].strip()
        return result


class MarianMTTranslator(BaseTranslator):
    """MarianMT 翻譯器"""

    def __init__(
        self, src_lang: str, tgt_lang: str, device: str = "mps", max_length: int = 1024
    ):
        super().__init__(src_lang, tgt_lang)
        self.device = device
        self.max_length = max_length
        self.translators = {}
        self.tgt_lang_for_model = None
        self.use_pivot = False

    def _get_translator(self, src: str, tgt: str):
        """獲取翻譯器"""
        from huggingface_hub.utils import RepositoryNotFoundError
        from transformers import pipeline
        from transformers.pipelines import Pipeline

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        try:
            return pipeline("translation", model=model_name, device=self.device)
        except (OSError, RepositoryNotFoundError):
            return None

    def _load_model(self) -> None:
        """載入 MarianMT 模型"""
        from huggingface_hub.utils import RepositoryNotFoundError
        from transformers.pipelines import Pipeline

        # 處理目標語言
        self.tgt_lang_for_model = self.tgt_lang
        if self.tgt_lang.lower() in ["zht", "zh", "zhs"]:
            self.tgt_lang_for_model = "zh"

        # 嘗試直接 src-tgt
        direct_translator = self._get_translator(self.src_lang, self.tgt_lang_for_model)
        if direct_translator:
            self.translators["direct"] = direct_translator
            print(f"載入直接翻譯模型: opus-mt-{self.src_lang}-{self.tgt_lang_for_model}")
            return

        # 若找不到，嘗試 src-en + en-tgt
        if self.src_lang != "en" and self.tgt_lang_for_model != "en":
            src2en = self._get_translator(self.src_lang, "en")
            en2tgt = self._get_translator("en", self.tgt_lang_for_model)
            if src2en and en2tgt:
                self.translators["src2en"] = src2en
                self.translators["en2tgt"] = en2tgt
                self.use_pivot = True
                print(
                    f"載入中轉翻譯模型: opus-mt-{self.src_lang}-en + opus-mt-en-{self.tgt_lang_for_model}"
                )
                return

        raise RuntimeError(f"找不到可用的 {self.src_lang}->{self.tgt_lang} 翻譯模型，也無法中轉。")

    def _translate_text(self, text: str) -> str:
        """使用 MarianMT 翻譯文字"""
        if "direct" in self.translators:
            translated = self.translators["direct"](text, max_length=self.max_length)[
                0
            ]["translation_text"].strip()
        elif "src2en" in self.translators and "en2tgt" in self.translators:
            en_text = self.translators["src2en"](text, max_length=self.max_length)[0][
                "translation_text"
            ].strip()
            translated = self.translators["en2tgt"](
                en_text, max_length=self.max_length
            )[0]["translation_text"].strip()
        else:
            raise RuntimeError("沒有可用的翻譯模型")

        return translated


def create_translator(
    translator_type: str, src_lang: str, tgt_lang: str, **kwargs
) -> BaseTranslator:
    """
    工廠函數：創建翻譯器

    Args:
        translator_type: 翻譯器類型 ("llama" 或 "marianmt")
        src_lang: 來源語言
        tgt_lang: 目標語言
        **kwargs: 其他參數

    Returns:
        翻譯器實例
    """
    if translator_type.lower() == "llama":
        return LlamaTranslator(src_lang, tgt_lang, **kwargs)
    elif translator_type.lower() == "marianmt":
        return MarianMTTranslator(src_lang, tgt_lang, **kwargs)
    else:
        raise ValueError(f"不支援的翻譯器類型: {translator_type}")
