#!/usr/bin/env python3
"""
使用 Objective-C 程式進行影片 OCR 並生成字幕檔
避免 Python PyObjC 橋接層的記憶體洩漏問題
"""

import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pysubs2

from macos_video_auto_ocr_ass.constants import (
    DEFAULT_DOWNSCALE,
    DEFAULT_FONT_SIZE,
    DEFAULT_INTERVAL,
    LOGGER_NAME,
    VIDEO_OCR_DEFAULT_CHUNK_SIZE,
    VIDEO_OCR_DEFAULT_DOWNSCALE_OBJC,
)
from macos_video_auto_ocr_ass.logger import get_logger

# Try to import tqdm for progress bar
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = get_logger(LOGGER_NAME)


def run_video_ocr_to_json(
    video_path: str,
    output_json: str,
    interval: float = DEFAULT_INTERVAL,
    downscale: int = VIDEO_OCR_DEFAULT_DOWNSCALE_OBJC,
    scan_rect: Optional[Tuple[int, int, int, int]] = None,
    languages: Optional[List[str]] = None,
    quiet: bool = False,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    show_progress: bool = False,
) -> None:
    """使用 Objective-C 程式進行影片 OCR 並輸出 JSON"""
    if not quiet:
        logger.debug(f"Using Objective-C OCR for {video_path}")

    ocr_path = "./video_ocr_to_json"
    if not os.path.exists(ocr_path):
        raise FileNotFoundError(f"Objective-C OCR tool not found: {ocr_path}")

    cmd = [ocr_path, video_path, output_json, str(interval), str(downscale)]

    if scan_rect:
        scan_rect_str = ",".join(map(str, scan_rect))
        cmd.append(scan_rect_str)
    elif len(cmd) == 5:
        cmd.append("0,0,1,1")

    if languages:
        if isinstance(languages, (list, tuple)):
            languages_str = ",".join(languages)
        else:
            languages_str = languages
        if len(cmd) == 5:
            cmd.append("0,0,1,1")
        cmd.append(languages_str)
    else:
        cmd.append("__AUTO__")

    if start_time is not None:
        cmd.append(str(start_time))
    if end_time is not None:
        cmd.append(str(end_time))
    cmd.append("1" if show_progress else "0")

    if not quiet:
        logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd)
        proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Objective-C OCR failed: {e}")
        raise


def add_ocr_to_subs(
    subs: Any,
    frame_data: Dict[str, Any],
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    x_offset: int = 0,
    base_font_size: int = DEFAULT_FONT_SIZE,
    quiet: bool = False,
    debug: bool = False,
) -> None:
    """將 OCR 結果添加到字幕檔案"""
    for style in subs.styles.values():
        style.alignment = 5
        style.marginl = 0
        style.marginr = 0
        style.marginv = 0
        style.borderstyle = 1
        style.outline = 0
        style.shadow = 0
        style.margin_t = 0
        style.margin_b = 0

    timestamp = frame_data["timestamp"]
    start = int(round(timestamp * 1000))
    end = int(round((timestamp + 1.0) * 1000))

    for result in frame_data["results"]:
        text = result["text"].strip()
        bbox = result["bbox"]

        if debug:
            logger.debug(f"bbox 原始值: {bbox}")

        if text:
            x_raw = bbox[0]
            y_raw = bbox[1]
            w_raw = bbox[2]
            h_raw = bbox[3]

            is_normalized = all(0.0 <= v <= 1.0 for v in [x_raw, y_raw, w_raw, h_raw])

            if is_normalized:
                center_x_norm = x_raw + w_raw / 2
                center_y_norm = y_raw + h_raw / 2
                x = int(center_x_norm * original_width) + x_offset
                y_vision = center_y_norm * original_height
                y = int(original_height - y_vision)
                if debug:
                    logger.debug(
                        f"bbox 為比例，轉換後 x={x}, y={y} (原始寬高 {original_width}x{original_height})"
                    )
            else:
                center_x = x_raw + w_raw / 2
                center_y = y_raw + h_raw / 2
                x = int(center_x) + x_offset
                y = int(original_height - center_y)
                if debug:
                    logger.debug(
                        f"bbox 為像素座標，直接使用 x={x}, y={y} (原始寬高 {original_width}x{original_height})"
                    )

            pos_tag = f"{{\\pos({x},{y})}}"
            if debug:
                logger.debug(f"產生 pos 標籤: {pos_tag} 文字: {text}")
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=f"{pos_tag}{text}"))
        elif text:
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))


def parse_pos(text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"\\pos\((\d+),(\d+)\)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def merge_ass_events(
    subs: Any,
    position_tolerance: int = 10,
    time_gap_threshold: int = 500,
    base_font_size: int = DEFAULT_FONT_SIZE,
) -> int:
    merged_events = []
    events = sorted(subs.events, key=lambda e: (e.text))
    if not events:
        return 0
    current = events[0]
    for next_event in events[1:]:
        cur_pos = parse_pos(current.text)
        next_pos = parse_pos(next_event.text)
        cur_text = current.text
        next_text = next_event.text
        pos_close = (
            cur_pos[0] is not None
            and next_pos[0] is not None
            and abs(cur_pos[0] - next_pos[0]) <= position_tolerance
            and abs(cur_pos[1] - next_pos[1]) <= position_tolerance
        )
        text_same = cur_text == next_text
        time_overlap_or_close = next_event.start <= current.end + time_gap_threshold
        if pos_close and text_same and time_overlap_or_close:
            current.end = max(current.end, next_event.end)
        else:
            merged_events.append(current)
            current = next_event
    merged_events.append(current)
    subs.events = merged_events
    subs.sort()
    return len(merged_events)


def get_video_duration(video_path: str) -> float:
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
    video_path: str,
    output_ass: str,
    interval: float = DEFAULT_INTERVAL,
    recognition_languages: Optional[List[str]] = None,
    quiet: bool = False,
    show_progress: bool = False,
    downscale: int = VIDEO_OCR_DEFAULT_DOWNSCALE_OBJC,
    chunk_size: float = VIDEO_OCR_DEFAULT_CHUNK_SIZE,
    x_offset: int = 0,
    scan_rect: Optional[Tuple[int, int, int, int]] = None,
    base_font_size: int = DEFAULT_FONT_SIZE,
    merge_events: bool = True,
    position_tolerance: int = 10,
    time_gap_threshold: int = 500,
) -> None:
    debug = not quiet
    if debug:
        logger.debug(f"Processing video: {video_path}")
        logger.debug(f"Output ASS: {output_ass}")

    duration = get_video_duration(video_path)
    if debug:
        logger.debug(f"Video duration: {duration} seconds")

    num_chunks = math.ceil(duration / chunk_size)
    temp_json_files = []
    for i in range(num_chunks):
        start_time = i * chunk_size
        end_time = min((i + 1) * chunk_size, duration)
        if start_time >= end_time:
            if not quiet:
                logger.debug(
                    f"Skip chunk {i}: start_time ({start_time}) >= end_time ({end_time})"
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

    ocr_results = []
    for json_file in temp_json_files:
        with open(json_file, "r") as f:
            ocr_results.extend(json.load(f))
        os.unlink(json_file)

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

    subs = pysubs2.SSAFile()
    style = pysubs2.SSAStyle(
        fontname="Arial",
        fontsize=base_font_size,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        outlinecolor=pysubs2.Color(0, 0, 0, 0),
        bold=True,
    )
    subs.styles["Default"] = style

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

    if merge_events:
        if not quiet:
            logger.info("Merging continuous events (pos+text+time)...")
        num_events = merge_ass_events(
            subs,
            position_tolerance=position_tolerance,
            time_gap_threshold=time_gap_threshold,
            base_font_size=base_font_size,
        )
        if not quiet:
            logger.info(f"Final number of events: {num_events}")

    subs.save(output_ass)
    if not quiet:
        logger.info(f"Subtitle file saved to: {output_ass}")


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
        default=DEFAULT_INTERVAL,
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
        default=VIDEO_OCR_DEFAULT_DOWNSCALE_OBJC,
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
        default=DEFAULT_FONT_SIZE,
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
        default=VIDEO_OCR_DEFAULT_CHUNK_SIZE,
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

    scan_rect = None
    if args.scan_rect:
        try:
            scan_rect = tuple(map(int, args.scan_rect.split(",")))
            if len(scan_rect) != 4:
                raise ValueError("Scan rect must have exactly 4 values")
            if not all(x >= 0 for x in scan_rect):
                raise ValueError("Scan rect values must be non-negative integers")
        except Exception as e:
            logger.error(f"Error parsing scan rect: {e}")
            sys.exit(1)

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
