import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import objc
import pysubs2
from AVFoundation import AVAsset, AVAssetImageGenerator
from Cocoa import NSURL
from CoreMedia import CMTimeMakeWithSeconds
from Foundation import NSValue
from PIL import Image
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CIImage,
)
from Vision import VNImageRequestHandler, VNRecognizeTextRequest

# Try to import tqdm for progress bar
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Helper to convert CGImageRef to PIL Image
def cgimage_to_pil(cg_image):
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    buffer = bytes(data)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    arr = arr.reshape((height, width, 4))
    return Image.fromarray(arr[..., :3])


def extract_and_downscale_frames(
    video_path, interval=1.0, downscale=2, quiet=False, scan_rect=None, debug=False
):
    if debug:
        print(f"[DEBUG] Entering extract_and_downscale_frames for {video_path}")
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    generator = AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
    generator.setAppliesPreferredTrackTransform_(True)
    duration = asset.duration().value / asset.duration().timescale
    if debug:
        print(f"[DEBUG] Video duration: {duration} seconds")
    times = [CMTimeMakeWithSeconds(t, 600) for t in np.arange(0, duration, interval)]
    yielded = False
    for idx, (t, cm_time) in enumerate(zip(np.arange(0, duration, interval), times)):
        if t >= duration:
            if debug:
                print(
                    f"[DEBUG] Skipping frame at {t:.2f}s (beyond duration {duration:.2f}s)"
                )
            continue
        if debug:
            print(f"[DEBUG] Attempting to extract frame at {t:.2f}s (index {idx})")
        try:
            result = generator.copyCGImageAtTime_actualTime_error_(cm_time, None, None)
            if isinstance(result, tuple):
                cg_image = result[0]
            else:
                cg_image = result
            if cg_image is None:
                if debug:
                    print(f"[DEBUG] cg_image is None at {t:.2f}s")
                continue
        except Exception as e:
            if debug:
                print(f"[DEBUG] Failed to extract frame at {t:.2f}s: {e}")
            continue
        pil_image = cgimage_to_pil(cg_image)
        if downscale and downscale > 1:
            pil_image = pil_image.resize(
                (pil_image.width // downscale, pil_image.height // downscale)
            )
        crop_offset = (0, 0)
        if scan_rect is not None:
            # scan_rect: (x, y, width, height) in original resolution
            x, y, w, h = scan_rect
            # 轉換到 downscaled 幀座標
            x_ = int(x / downscale)
            y_ = int(y / downscale)
            w_ = int(w / downscale)
            h_ = int(h / downscale)
            pil_image = pil_image.crop((x_, y_, x_ + w_, y_ + h_))
            crop_offset = (x_, y_)
        if debug:
            print(
                f"[DEBUG] Extracted and downscaled frame at {t:.2f}s, PIL image size: {getattr(pil_image, 'size', None)}, mode: {getattr(pil_image, 'mode', None)}, crop_offset: {crop_offset}"
            )
        yielded = True
        yield t, pil_image, crop_offset
    if not yielded and debug:
        print("[DEBUG] No frames were yielded from extract_and_downscale_frames!")


def ocr_image(image, recognition_languages=None, quiet=False, debug=False):
    if debug:
        print(
            f"[DEBUG] Entering ocr_image, image type: {type(image)}, size: {getattr(image, 'size', None)}"
        )
    import io

    from Cocoa import NSData

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    nsdata = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    ci_image = CIImage.imageWithData_(nsdata)
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    results = []

    def handler_block(request, error):
        if error is not None:
            return
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            text = candidates[0].string() if candidates and candidates[0] else ""
            bbox = obs.boundingBox() if hasattr(obs, "boundingBox") else None
            results.append((text, bbox))

    request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler_block)
    # 預設啟用自動語言偵測
    request.setAutomaticallyDetectsLanguage_(True)
    if recognition_languages:
        if debug:
            print(f"[DEBUG] Setting recognition languages: {recognition_languages}")
        request.setRecognitionLanguages_(recognition_languages)
    handler.performRequests_error_([request], None)
    return results


def add_ocr_to_subs(
    subs,
    t,
    frame,
    ocr_results,
    interval,
    x_offset=0,
    original_width=None,
    original_height=None,
    downscale=1,
    base_font_size=24,
    quiet=False,
    debug=False,
):
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
    start = int(round(t * 1000))
    end = int(round((t + interval) * 1000))
    frame_width, frame_height = frame.size

    # 如果沒有提供原始分辨率，使用幀的分辨率
    if original_width is None:
        original_width = frame_width
    if original_height is None:
        original_height = frame_height

    # 計算字體大小（根據 downscale 比例調整）
    font_size = int(base_font_size * downscale)

    for text, bbox in ocr_results:
        text = text.strip()
        if text and bbox is not None:
            # Vision: 左下為原點, 歸一化 (0-1); ASS: 左上為原點, 使用 PlayRes 座標系統
            x_norm = bbox.origin.x
            y_norm = bbox.origin.y
            w_norm = bbox.size.width
            h_norm = bbox.size.height

            # 計算 Vision 框中心點（歸一化座標）
            center_x_norm = x_norm + w_norm / 2
            center_y_norm = y_norm + h_norm / 2

            # 轉換到原始視頻分辨率（考慮 downscaling）
            # 由於 Vision 的歸一化是基於輸入圖像的，而輸入圖像是 downscaled 的
            # 所以我們需要先轉換到 downscaled 座標，再轉換到原始座標
            x_crop = center_x_norm * frame_width
            y_crop = center_y_norm * frame_height

            # 轉換到原始視頻分辨率
            x = int(x_crop * downscale) + x_offset
            y_vision = y_crop * downscale  # Vision: 下到上
            y = int(original_height - y_vision)  # ASS: 上到下

            pos_tag = f"{{\\pos({x},{y})\\fs{font_size}}}"
            if debug:
                print(
                    f"[DEBUG] OCR text '{text}' at normalized bbox: x={x_norm:.3f}, y={y_norm:.3f}, w={w_norm:.3f}, h={h_norm:.3f}"
                )
                print(
                    f"[DEBUG] Frame size: {frame_width}x{frame_height}, Original size: {original_width}x{original_height}, Downscale: {downscale}"
                )
                print(
                    f"[DEBUG] Converted to pixel position: x={x} (with offset {x_offset}), y={y}, font_size: {font_size}"
                )
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=f"{pos_tag}{text}"))
        elif text:
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))


def merge_continuous_events(subs, position_tolerance=10, time_gap_threshold=500):
    """
    合併連續的相同文字事件
    """
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
    downscale=2,
    chunk_size=None,  # ignored, for compatibility
    x_offset=0,  # X 軸偏移補償
    scan_rect=None,  # 掃描區域 (x, y, w, h)
    base_font_size=24,  # 基礎字體大小
    merge_events=True,  # 是否合併連續事件
    position_tolerance=10,  # 位置容差（像素）
    time_gap_threshold=500,  # 時間間隙閾值（毫秒）
):
    debug = not quiet
    if debug:
        print(
            f"[DEBUG] Starting main with video_path={video_path}, output_ass={output_ass}, interval={interval}, recognition_languages={recognition_languages}, downscale={downscale}, x_offset={x_offset}, base_font_size={base_font_size}, merge_events={merge_events}"
        )
    subs = pysubs2.SSAFile()

    # 設置默認樣式
    default_style = pysubs2.SSAStyle()
    default_style.alignment = 5  # 中心對齊
    default_style.marginl = 0
    default_style.marginr = 0
    default_style.marginv = 0
    default_style.borderstyle = 1
    default_style.outline = 0
    default_style.shadow = 0
    default_style.margin_t = 0
    default_style.margin_b = 0
    subs.styles["Default"] = default_style

    # Get total frames for progress bar
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    duration = asset.duration().value / asset.duration().timescale
    total_frames = int(duration // interval)

    # 獲取視頻的實際分辨率並設置 PlayRes
    original_width = None
    original_height = None
    tracks = asset.tracks()
    if tracks and len(tracks) > 0:
        video_track = None
        for track in tracks:
            if track.mediaType() == "vide":
                video_track = track
                break
        if video_track:
            natural_size = video_track.naturalSize()
            original_width = int(natural_size.width)
            original_height = int(natural_size.height)
            if not quiet:
                print(
                    f"[DEBUG] Video natural size: {original_width} x {original_height}"
                )
            # 設置 PlayRes 為視頻的實際分辨率
            subs.info["PlayResX"] = original_width
            subs.info["PlayResY"] = original_height

    frame_iter = extract_and_downscale_frames(
        video_path, interval, downscale, quiet=quiet, scan_rect=scan_rect, debug=debug
    )
    if HAS_TQDM:
        frame_iter = tqdm(frame_iter, total=total_frames, desc="Frames")
    frame_count = 0
    for t, frame, crop_offset in frame_iter:
        frame_count += 1
        ocr_results = ocr_image(
            frame, recognition_languages=recognition_languages, quiet=quiet, debug=debug
        )
        if debug:
            print(f"[DEBUG] OCR at {t:.2f}s: {repr(ocr_results)}")
        if ocr_results:
            add_ocr_to_subs(
                subs,
                t,
                frame,
                ocr_results,
                interval,
                x_offset,
                original_width,
                original_height,
                downscale,
                base_font_size,
                quiet,
                debug,
            )
        del frame
        gc.collect()
    if frame_count == 0 and debug:
        print("[DEBUG] No frames processed!")

    # 合併連續事件
    if merge_events and subs.events:
        original_count = len(subs.events)
        merged_count = merge_continuous_events(
            subs, position_tolerance, time_gap_threshold
        )
        if not quiet:
            print(f"[DEBUG] Merged events: {original_count} -> {merged_count}")

    subs.sort()
    subs.save(output_ass)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video OCR to .ass subtitle file with language and progress options."
    )
    parser.add_argument("input_video", help="Input video file")
    parser.add_argument("output_ass", help="Output .ass subtitle file")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Frame interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated OCR languages, e.g. ko,en",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress debug messages")
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="Downscale factor for frames before OCR (default: 2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50, help="(ignored, for compatibility)"
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=0,
        help="X-axis offset compensation in pixels (default: 0)",
    )
    parser.add_argument(
        "--scan-rect",
        type=str,
        default=None,
        help="Scan region in format x,y,width,height (in original resolution, left-top origin)",
    )
    parser.add_argument(
        "--base-font-size",
        type=int,
        default=24,
        help="Base font size (will be multiplied by downscale factor) (default: 24)",
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="Disable merging of continuous events"
    )
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=10,
        help="Position tolerance for merging (pixels) (default: 10)",
    )
    parser.add_argument(
        "--time-gap-threshold",
        type=int,
        default=500,
        help="Max time gap (ms) to merge (default: 500)",
    )
    args = parser.parse_args()
    recognition_languages = args.languages.split(",") if args.languages else None
    scan_rect = None
    if args.scan_rect:
        try:
            scan_rect = tuple(map(int, args.scan_rect.split(",")))
            assert len(scan_rect) == 4
        except Exception:
            raise ValueError(
                "--scan-rect must be in format x,y,width,height (all integers)"
            )
    main(
        args.input_video,
        args.output_ass,
        args.interval,
        recognition_languages,
        quiet=args.quiet,
        downscale=args.downscale,
        chunk_size=args.chunk_size,
        x_offset=args.x_offset,
        scan_rect=scan_rect,
        base_font_size=args.base_font_size,
        merge_events=not args.no_merge,
        position_tolerance=args.position_tolerance,
        time_gap_threshold=args.time_gap_threshold,
    )
