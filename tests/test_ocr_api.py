import os

import requests

API_URL = "http://localhost:5001"

# 測試用影片與字幕檔案路徑（請自行替換成實際檔案）
TEST_VIDEO = "/path/to/test.mp4"
TEST_ASS = "/path/to/test.ass"


def test_ocr_ass():
    if not os.path.exists(TEST_VIDEO):
        print("請先準備測試影片:", TEST_VIDEO)
        return
    resp = requests.post(
        f"{API_URL}/ocr_ass",
        json={
            "video_path": TEST_VIDEO,
            "interval": 1.0,
            "languages": "zh-Hant",
            "downscale": 2,
            "base_font_size": 24,
            "quiet": True,
        },
    )
    print("/ocr_ass:", resp.status_code)
    print(resp.json())


def test_ocr_ass_objc():
    if not os.path.exists(TEST_VIDEO):
        print("請先準備測試影片:", TEST_VIDEO)
        return
    resp = requests.post(
        f"{API_URL}/ocr_ass_objc",
        json={
            "video_path": TEST_VIDEO,
            "interval": 1.0,
            "languages": "zh-Hant",
            "downscale": 1,
            "base_font_size": 24,
            "quiet": True,
        },
    )
    print("/ocr_ass_objc:", resp.status_code)
    print(resp.json())


def test_merge_ass():
    if not os.path.exists(TEST_ASS):
        print("請先準備測試字幕:", TEST_ASS)
        return
    with open(TEST_ASS, "r", encoding="utf-8") as f:
        ass_content = f.read()
    resp = requests.post(
        f"{API_URL}/merge_ass",
        json={
            "ass_content": ass_content,
            "position_tolerance": 10,
            "time_gap_threshold": 500,
            "base_font_size": 24,
        },
    )
    print("/merge_ass:", resp.status_code)
    print(resp.json())


def test_translate_ass():
    if not os.path.exists(TEST_ASS):
        print("請先準備測試字幕:", TEST_ASS)
        return
    with open(TEST_ASS, "r", encoding="utf-8") as f:
        ass_content = f.read()
    resp = requests.post(
        f"{API_URL}/translate_ass",
        json={
            "ass_content": ass_content,
            "src_lang": "en",
            "tgt_lang": "zh",
            "device": "cpu",
            "show_text": False,
        },
    )
    print("/translate_ass:", resp.status_code)
    print(resp.json())


if __name__ == "__main__":
    test_ocr_ass()
    test_ocr_ass_objc()
    test_merge_ass()
    test_translate_ass()
