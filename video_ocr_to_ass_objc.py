#!/usr/bin/env python3
"""
使用 Objective-C 程式進行影片 OCR 並生成字幕檔
避免 Python PyObjC 橋接層的記憶體洩漏問題
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pysubs2

# Try to import tqdm for progress bar
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_video_ocr_to_json(
    video_path,
    output_json,
    interval=1.0,
    downscale=1,
    scan_rect=None,
    languages=None,
    quiet=False,
):
    """使用 Objective-C 程式進行影片 OCR 並輸出 JSON"""
    if not quiet:
        print(f"[DEBUG] Using Objective-C OCR for {video_path}")

    # 確保 video_ocr_to_json 可執行檔存在
    ocr_path = "./video_ocr_to_json"
    if not os.path.exists(ocr_path):
        raise FileNotFoundError(f"Objective-C OCR tool not found: {ocr_path}")

    # 建立命令
    cmd = [ocr_path, video_path, output_json, str(interval), str(downscale)]

    # 添加掃描區域參數（如果有）
    if scan_rect:
        scan_rect_str = ",".join(map(str, scan_rect))
        cmd.append(scan_rect_str)
    elif len(cmd) == 5:  # 如果有 downscale 但沒有 scan_rect，需要添加預設值
        cmd.append("0,0,1,1")

    # 添加語言參數（如果有）
    if languages:
        if isinstance(languages, (list, tuple)):
            languages_str = ",".join(languages)
        else:
            languages_str = languages
        if len(cmd) == 5:  # 需要先添加預設的 scan_rect
            cmd.append("0,0,1,1")
        cmd.append(languages_str)

    if not quiet:
        print(f"[DEBUG] Running command: {' '.join(cmd)}")

    # 執行 Objective-C 程式
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not quiet:
            print(f"[DEBUG] Objective-C OCR output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Objective-C OCR failed: {e}")
        print(f"[ERROR] stderr: {e.stderr}")
        raise


def add_ocr_to_subs(
    subs,
    frame_data,
    original_width=None,
    original_height=None,
    x_offset=0,
    base_font_size=24,
    quiet=False,
    debug=False,
):
    """將 OCR 結果添加到字幕檔案"""
    # 強制所有 Style 的 Alignment 為 5 (中心)
    for style in subs.styles.values():
        style.alignment = 5
        style.marginl = 0
        style.marginr = 0
        style.marginv = 0
        # 添加更多樣式設置以確保位置準確
        style.borderstyle = 1  # 1 = 外邊框
        style.outline = 0  # 外邊框寬度
        style.shadow = 0  # 陰影寬度
        style.margin_t = 0  # 頂部邊距
        style.margin_b = 0  # 底部邊距

    timestamp = frame_data["timestamp"]
    start = int(round(timestamp * 1000))
    end = int(round((timestamp + 1.0) * 1000))  # 假設間隔為 1 秒

    for result in frame_data["results"]:
        text = result["text"].strip()
        bbox = result["bbox"]  # [x, y, width, height]

        # Debug print bbox 原始值
        if debug:
            print(f"[DEBUG] bbox 原始值: {bbox}")

        if text:
            x_raw = bbox[0]
            y_raw = bbox[1]
            w_raw = bbox[2]
            h_raw = bbox[3]

            # 判斷 bbox 是否為 0~1 區間（比例），否則視為像素座標
            is_normalized = all(0.0 <= v <= 1.0 for v in [x_raw, y_raw, w_raw, h_raw])

            if is_normalized:
                # 歸一化比例，需乘原始解析度
                center_x_norm = x_raw + w_raw / 2
                center_y_norm = y_raw + h_raw / 2
                x = int(center_x_norm * original_width) + x_offset
                y_vision = center_y_norm * original_height
                y = int(original_height - y_vision)
                if debug:
                    print(
                        f"[DEBUG] bbox 為比例，轉換後 x={x}, y={y} (原始寬高 {original_width}x{original_height})"
                    )
            else:
                # 已是像素座標，直接使用
                center_x = x_raw + w_raw / 2
                center_y = y_raw + h_raw / 2
                x = int(center_x) + x_offset
                y = int(original_height - center_y)
                if debug:
                    print(
                        f"[DEBUG] bbox 為像素座標，直接使用 x={x}, y={y} (原始寬高 {original_width}x{original_height})"
                    )

            pos_tag = f"{{\\pos({x},{y})}}"
            if debug:
                print(f"[DEBUG] 產生 pos 標籤: {pos_tag} 文字: {text}")
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=f"{pos_tag}{text}"))
        elif text:
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))


def merge_continuous_events(subs, position_tolerance=10, time_gap_threshold=500):
    """合併連續的相同文字事件"""
    if not subs.events:
        return

    # 按時間排序
    subs.events.sort(key=lambda x: x.start)

    merged_events = []
    current_event = subs.events[0].copy()

    for next_event in subs.events[1:]:
        # 檢查是否為相同文字且時間連續
        if (
            current_event.text == next_event.text
            and current_event.end >= next_event.start - time_gap_threshold
        ):  # 允許指定間隙
            # 延長當前事件的結束時間
            current_event.end = max(current_event.end, next_event.end)
        else:
            # 添加當前事件並開始新事件
            merged_events.append(current_event)
            current_event = next_event.copy()

    # 添加最後一個事件
    merged_events.append(current_event)

    # 更新字幕檔案的事件
    subs.events = merged_events
    return len(subs.events)


def main(
    video_path,
    output_ass,
    interval=1.0,
    recognition_languages=None,
    quiet=False,
    downscale=1,
    chunk_size=None,  # ignored, for compatibility
    x_offset=0,  # X 軸偏移補償
    scan_rect=None,  # 掃描區域 (x, y, w, h)，需要是 0-1 之間的值
    base_font_size=24,  # 基礎字體大小
    merge_events=True,  # 是否合併連續事件
    position_tolerance=10,  # 位置容差（像素）
    time_gap_threshold=500,  # 時間間隙閾值（毫秒）
):
    debug = not quiet
    if debug:
        print(f"[DEBUG] Processing video: {video_path}")
        print(f"[DEBUG] Output ASS: {output_ass}")

    # 建立臨時 JSON 檔案
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_json = tmp.name

    try:
        # 執行 OCR 並生成 JSON
        run_video_ocr_to_json(
            video_path,
            temp_json,
            interval=interval,
            downscale=downscale,
            scan_rect=scan_rect,
            languages=recognition_languages,
            quiet=quiet,
        )

        # 讀取 JSON 結果
        with open(temp_json, "r") as f:
            ocr_results = json.load(f)

        # 獲取影片尺寸
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        original_width = video_info["streams"][0]["width"]
        original_height = video_info["streams"][0]["height"]

        # 建立字幕檔案
        subs = pysubs2.SSAFile()
        style = pysubs2.SSAStyle(
            fontname="Arial",
            fontsize=base_font_size,
            primarycolor=pysubs2.Color(255, 255, 255, 0),  # 白色
            outlinecolor=pysubs2.Color(0, 0, 0, 0),  # 黑色邊框
            bold=True,
        )
        subs.styles["Default"] = style

        # 處理每一幀的 OCR 結果
        if HAS_TQDM and not quiet:
            ocr_results = tqdm(ocr_results, desc="Processing frames")

        for frame_data in ocr_results:
            add_ocr_to_subs(
                subs,
                frame_data,
                original_width=original_width,
                original_height=original_height,
                x_offset=x_offset,
                base_font_size=base_font_size,
                quiet=quiet,
                debug=debug,
            )

        # 合併連續事件
        if merge_events:
            if not quiet:
                print("[INFO] Merging continuous events...")
            num_events = merge_continuous_events(
                subs,
                position_tolerance=position_tolerance,
                time_gap_threshold=time_gap_threshold,
            )
            if not quiet:
                print(f"[INFO] Final number of events: {num_events}")

        # 保存字幕檔案
        subs.save(output_ass)
        if not quiet:
            print(f"[INFO] Subtitle file saved to: {output_ass}")

    finally:
        # 清理臨時檔案
        try:
            os.unlink(temp_json)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ASS subtitles from video using OCR"
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_ass", help="Path to output ASS subtitle file")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Time interval between frames in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated OCR languages (e.g., ja,en,zh-Hant)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=1,
        help="Downscale factor for frames before OCR (default: 1)",
    )
    parser.add_argument(
        "--scan-rect",
        type=str,
        default=None,
        help="Scan region in format x,y,width,height (normalized 0-1, default: entire frame)",
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=0,
        help="X-axis offset compensation in pixels (default: 0)",
    )
    parser.add_argument(
        "--base-font-size",
        type=int,
        default=24,
        help="Base font size for subtitles (default: 24)",
    )
    parser.add_argument(
        "--no-merge-events",
        action="store_false",
        dest="merge_events",
        help="Disable merging of continuous events",
    )
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=10,
        help="Position tolerance in pixels for merging events (default: 10)",
    )
    parser.add_argument(
        "--time-gap-threshold",
        type=int,
        default=500,
        help="Time gap threshold in milliseconds for merging events (default: 500)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress debug output",
    )

    args = parser.parse_args()

    # 解析掃描區域
    scan_rect = None
    if args.scan_rect:
        try:
            scan_rect = tuple(map(int, args.scan_rect.split(",")))
            if len(scan_rect) != 4:
                raise ValueError("Scan rect must have exactly 4 values")
            # 檢查是否為有效的像素座標（非負整數）
            if not all(x >= 0 for x in scan_rect):
                raise ValueError("Scan rect values must be non-negative integers")
        except Exception as e:
            print(f"Error parsing scan rect: {e}")
            sys.exit(1)

    # 解析語言設定
    recognition_languages = args.languages.split(",") if args.languages else None

    main(
        video_path=args.video_path,
        output_ass=args.output_ass,
        interval=args.interval,
        recognition_languages=recognition_languages,
        quiet=args.quiet,
        downscale=args.downscale,
        x_offset=args.x_offset,
        scan_rect=scan_rect,
        base_font_size=args.base_font_size,
        merge_events=args.merge_events,
        position_tolerance=args.position_tolerance,
        time_gap_threshold=args.time_gap_threshold,
    )
