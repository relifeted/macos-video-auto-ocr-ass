#!/usr/bin/env python3
"""
使用 Objective-C 程式進行影片 OCR 並生成字幕檔
避免 Python PyObjC 橋接層的記憶體洩漏問題
"""

import json
import math
import multiprocessing
import os
import shutil
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
    start_time=None,
    end_time=None,
    show_progress=False,
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
    else:
        cmd.append("__AUTO__")  # 傳遞特殊字串，代表自動偵測語言

    # 新增 start_time, end_time, show_progress
    if start_time is not None:
        cmd.append(str(start_time))
    if end_time is not None:
        cmd.append(str(end_time))
    cmd.append("1" if show_progress else "0")

    if not quiet:
        print(f"[DEBUG] Running command: {' '.join(cmd)}")

    # 改用 Popen 直接 relay stdout/stderr
    try:
        proc = subprocess.Popen(cmd)
        proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Objective-C OCR failed: {e}")
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


def get_video_duration(video_path):
    """取得影片長度（秒）"""
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def main(
    video_path,
    output_ass,
    interval=1.0,
    recognition_languages=None,
    quiet=False,
    show_progress=False,
    downscale=1,
    chunk_size=300.0,  # 每段長度（秒）
    x_offset=0,
    scan_rect=None,
    base_font_size=24,
    merge_events=True,
    position_tolerance=10,
    time_gap_threshold=500,
):
    debug = not quiet
    if debug:
        print(f"[DEBUG] Processing video: {video_path}")
        print(f"[DEBUG] Output ASS: {output_ass}")

    # 取得影片長度
    duration = get_video_duration(video_path)
    if debug:
        print(f"[DEBUG] Video duration: {duration} seconds")

    # 分段
    num_chunks = math.ceil(duration / chunk_size)
    temp_json_files = []
    for i in range(num_chunks):
        start_time = i * chunk_size
        end_time = min((i + 1) * chunk_size, duration)
        if start_time >= end_time:
            if not quiet:
                print(
                    f"[DEBUG] Skip chunk {i}: start_time ({start_time}) >= end_time ({end_time})"
                )
            continue
        temp_json = tempfile.NamedTemporaryFile(suffix=f"_chunk{i}.json", delete=False)
        temp_json_files.append(temp_json.name)
        temp_json.close()
        run_video_ocr_to_json(
            video_path,
            temp_json_files[-1],
            interval=interval,
            downscale=downscale,
            scan_rect=scan_rect,
            languages=recognition_languages,
            quiet=quiet,
            start_time=start_time,
            end_time=end_time,
            show_progress=show_progress,
        )

    # 合併所有 JSON
    ocr_results = []
    for json_file in temp_json_files:
        with open(json_file, "r") as f:
            ocr_results.extend(json.load(f))
        os.unlink(json_file)

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
        "--chunk-size",
        type=float,
        default=300.0,
        help="Chunk size in seconds for parallel OCR (default: 300.0)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress debug output",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress for each chunk (default: False)",
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
        show_progress=args.show_progress,
        downscale=args.downscale,
        x_offset=args.x_offset,
        scan_rect=scan_rect,
        base_font_size=args.base_font_size,
        merge_events=args.merge_events,
        position_tolerance=args.position_tolerance,
        time_gap_threshold=args.time_gap_threshold,
        chunk_size=args.chunk_size,
    )
