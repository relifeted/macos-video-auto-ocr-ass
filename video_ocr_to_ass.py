import os
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import objc
import pysubs2
from AVFoundation import AVAsset, AVAssetImageGenerator
from Cocoa import NSURL
from CoreMedia import CMTimeMakeWithSeconds
from PIL import Image
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CIImage,
)
from Vision import VNImageRequestHandler, VNRecognizeTextRequest

from macos_video_auto_ocr_ass.constants import (
    CM_TIME_SCALE,
    DEFAULT_DOWNSCALE,
    DEFAULT_FONT_SIZE,
    DEFAULT_INTERVAL,
    LOGGER_NAME,
    MILLISECONDS_PER_SECOND,
)
from macos_video_auto_ocr_ass.logger import get_logger

# Try to import tqdm for progress bar
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = get_logger(LOGGER_NAME)


# Helper to convert CGImageRef to PIL Image
def cgimage_to_pil(cg_image: Any) -> Image.Image:
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    buffer = bytes(data)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    arr = arr.reshape((height, width, 4))
    return Image.fromarray(arr[..., :3])


def extract_and_downscale_frames(
    video_path: str,
    interval: float = DEFAULT_INTERVAL,
    downscale: int = DEFAULT_DOWNSCALE,
    quiet: bool = False,
    scan_rect: Optional[Tuple[int, int, int, int]] = None,
    debug: bool = False,
) -> Generator[Tuple[float, Image.Image, Tuple[int, int]], None, None]:
    if debug:
        logger.debug(f"Entering extract_and_downscale_frames for {video_path}")
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    temp_asset = AVAsset.assetWithURL_(url)
    duration = temp_asset.duration().value / temp_asset.duration().timescale
    del temp_asset
    if debug:
        logger.debug(f"Video duration: {duration} seconds")
    times = [
        CMTimeMakeWithSeconds(t, CM_TIME_SCALE)
        for t in np.arange(0, duration, interval)
    ]
    yielded = False
    for idx, (t, cm_time) in enumerate(zip(np.arange(0, duration, interval), times)):
        if t >= duration:
            if debug:
                logger.debug(
                    f"Skipping frame at {t:.2f}s (beyond duration {duration:.2f}s)"
                )
            continue
        if debug:
            logger.debug(f"Attempting to extract frame at {t:.2f}s (index {idx})")
        try:
            asset = AVAsset.assetWithURL_(url)
            generator = AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
            generator.setAppliesPreferredTrackTransform_(True)
            result = generator.copyCGImageAtTime_actualTime_error_(cm_time, None, None)
            if isinstance(result, tuple):
                cg_image = result[0]
            else:
                cg_image = result
            if cg_image is None:
                if debug:
                    logger.debug(f"cg_image is None at {t:.2f}s")
                del generator
                del asset
                continue
        except Exception as e:
            if debug:
                logger.debug(f"Failed to extract frame at {t:.2f}s: {e}")
            continue
        pil_image = cgimage_to_pil(cg_image)
        if downscale and downscale > 1:
            pil_image = pil_image.resize(
                (pil_image.width // downscale, pil_image.height // downscale)
            )
        crop_offset = (0, 0)
        if scan_rect is not None:
            x, y, w, h = scan_rect
            x_ = int(x / downscale)
            y_ = int(y / downscale)
            w_ = int(w / downscale)
            h_ = int(h / downscale)
            pil_image = pil_image.crop((x_, y_, x_ + w_, y_ + h_))
            crop_offset = (x_, y_)
        if debug:
            logger.debug(
                f"Extracted and downscaled frame at {t:.2f}s, PIL image size: {getattr(pil_image, 'size', None)}, mode: {getattr(pil_image, 'mode', None)}, crop_offset: {crop_offset}"
            )
        yielded = True
        yield t, pil_image, crop_offset
        del generator
        del asset
    if not yielded and debug:
        logger.debug("No frames were yielded from extract_and_downscale_frames!")


def ocr_image(
    image: Image.Image,
    recognition_languages: Optional[List[str]] = None,
    quiet: bool = False,
    debug: bool = False,
) -> List[Tuple[str, Any]]:
    if debug:
        logger.debug(
            f"Entering ocr_image, image type: {type(image)}, size: {getattr(image, 'size', None)}"
        )
    import io

    from Cocoa import NSData

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    nsdata = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    ci_image = CIImage.imageWithData_(nsdata)
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    results = []

    def handler_block(request: Any, error: Any) -> None:
        if error is not None:
            return
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            text = candidates[0].string() if candidates and candidates[0] else ""
            bbox = obs.boundingBox() if hasattr(obs, "boundingBox") else None
            results.append((text, bbox))

    request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler_block)
    request.setAutomaticallyDetectsLanguage_(True)
    if recognition_languages:
        if debug:
            logger.debug(f"Setting recognition languages: {recognition_languages}")
        request.setRecognitionLanguages_(recognition_languages)
    handler.performRequests_error_([request], None)
    return results


def add_ocr_to_subs(
    subs: Any,
    t: float,
    frame: Image.Image,
    ocr_results: List[Tuple[str, Any]],
    interval: float,
    x_offset: int = 0,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    downscale: int = 1,
    base_font_size: int = DEFAULT_FONT_SIZE,
    quiet: bool = False,
    debug: bool = False,
) -> None:
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
    start = int(round(t * MILLISECONDS_PER_SECOND))
    end = int(round((t + interval) * MILLISECONDS_PER_SECOND))
    frame_width, frame_height = frame.size

    if original_width is None:
        original_width = frame_width
    if original_height is None:
        original_height = frame_height

    font_size = int(base_font_size * downscale)

    for text, bbox in ocr_results:
        text = text.strip()
        if text and bbox is not None:
            x_norm = bbox.origin.x
            y_norm = bbox.origin.y
            w_norm = bbox.size.width
            h_norm = bbox.size.height

            center_x_norm = x_norm + w_norm / 2
            center_y_norm = y_norm + h_norm / 2

            x_crop = center_x_norm * frame_width
            y_crop = center_y_norm * frame_height

            x = int(x_crop * downscale) + x_offset
            y_vision = y_crop * downscale
            y = int(original_height - y_vision)

            pos_tag = f"{{\\pos({x},{y})\\fs{font_size}}}"
            if debug:
                logger.debug(
                    f"OCR text '{text}' at normalized bbox: x={x_norm:.3f}, y={y_norm:.3f}, w={w_norm:.3f}, h={h_norm:.3f}"
                )
                logger.debug(
                    f"Frame size: {frame_width}x{frame_height}, Original size: {original_width}x{original_height}, Downscale: {downscale}"
                )
                logger.debug(
                    f"Converted to pixel position: x={x} (with offset {x_offset}), y={y}, font_size: {font_size}"
                )
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=f"{pos_tag}{text}"))
        elif text:
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))


def merge_continuous_events(
    subs: Any, position_tolerance: int = 10, time_gap_threshold: int = 500
) -> int:
    """合併連續的相同文字事件"""
    if not subs.events:
        return 0

    subs.events.sort(key=lambda x: x.start)

    merged_events = []
    current_event = subs.events[0].copy()

    for next_event in subs.events[1:]:
        if (
            current_event.text == next_event.text
            and current_event.end >= next_event.start - time_gap_threshold
        ):
            current_event.end = max(current_event.end, next_event.end)
        else:
            merged_events.append(current_event)
            current_event = next_event.copy()

    merged_events.append(current_event)
    subs.events = merged_events
    return len(subs.events)


def main(
    video_path: str,
    output_ass: str,
    interval: float = DEFAULT_INTERVAL,
    recognition_languages: Optional[List[str]] = None,
    quiet: bool = False,
    downscale: int = DEFAULT_DOWNSCALE,
    chunk_size: Optional[float] = None,  # ignored, for compatibility
    x_offset: int = 0,  # X 軸偏移補償
    scan_rect: Optional[Tuple[int, int, int, int]] = None,  # 掃描區域 (x, y, w, h)
    base_font_size: int = DEFAULT_FONT_SIZE,  # 基礎字體大小
    merge_events: bool = True,  # 是否合併連續事件
    position_tolerance: int = 10,  # 位置容差（像素）
    time_gap_threshold: int = 500,  # 時間間隙閾值（毫秒）
) -> None:
    debug = not quiet
    if debug:
        logger.debug(
            f"Starting main with video_path={video_path}, output_ass={output_ass}, interval={interval}, recognition_languages={recognition_languages}, downscale={downscale}, x_offset={x_offset}, base_font_size={base_font_size}, merge_events={merge_events}"
        )
    subs = pysubs2.SSAFile()

    default_style = pysubs2.SSAStyle()
    default_style.alignment = 5
    default_style.marginl = 0
    default_style.marginr = 0
    default_style.marginv = 0
    default_style.borderstyle = 1
    default_style.outline = 0
    default_style.shadow = 0
    default_style.margin_t = 0
    default_style.margin_b = 0
    subs.styles["Default"] = default_style

    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    duration = asset.duration().value / asset.duration().timescale
    total_frames = int(duration // interval)

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
                logger.debug(
                    f"Video natural size: {original_width} x {original_height}"
                )
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
            logger.debug(f"OCR at {t:.2f}s: {repr(ocr_results)}")
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
    if frame_count == 0 and debug:
        logger.debug("No frames processed!")

    if merge_events and subs.events:
        original_count = len(subs.events)
        merged_count = merge_continuous_events(
            subs, position_tolerance, time_gap_threshold
        )
        if not quiet:
            logger.debug(f"Merged events: {original_count} -> {merged_count}")

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
        default=DEFAULT_INTERVAL,
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
        default=DEFAULT_DOWNSCALE,
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
        default=DEFAULT_FONT_SIZE,
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
