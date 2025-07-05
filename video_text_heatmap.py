import argparse
import os
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
from AVFoundation import AVAsset, AVAssetImageGenerator
from Cocoa import NSURL
from CoreMedia import CMTimeMakeWithSeconds
from PIL import Image, ImageDraw, ImageFont
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CIImage,
)
from Vision import VNImageRequestHandler, VNRecognizeTextRequest

from macos_video_auto_ocr_ass.constants import (
    DEFAULT_DOWNSCALE,
    DEFAULT_INTERVAL,
    HEATMAP_ALPHA_MULTIPLIER,
    HEATMAP_DEFAULT_CONTRAST_BOOST,
    HEATMAP_DEFAULT_FONT_SIZE,
    HEATMAP_DEFAULT_GRID_INTERVAL,
    HEATMAP_GREEN_MULTIPLIER,
    HEATMAP_GRID_COLOR,
    HEATMAP_LABEL_BG_COLOR,
    HEATMAP_LABEL_TEXT_COLOR,
    HEATMAP_WHITE_BG_COLOR,
    LOGGER_NAME,
)
from macos_video_auto_ocr_ass.logger import get_logger

# 嘗試匯入 tqdm
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


def extract_frames(
    video_path: str,
    interval: float = DEFAULT_INTERVAL,
    downscale: int = DEFAULT_DOWNSCALE,
    quiet: bool = False,
) -> Generator[Image.Image, None, None]:
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    generator = AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
    generator.setAppliesPreferredTrackTransform_(True)
    duration = asset.duration().value / asset.duration().timescale
    times = [CMTimeMakeWithSeconds(t, 600) for t in np.arange(0, duration, interval)]
    for idx, (t, cm_time) in enumerate(zip(np.arange(0, duration, interval), times)):
        if t >= duration:
            continue
        try:
            result = generator.copyCGImageAtTime_actualTime_error_(cm_time, None, None)
            cg_image = result[0] if isinstance(result, tuple) else result
            if cg_image is None:
                continue
        except Exception as e:
            if not quiet:
                logger.debug(f"Failed to extract frame at {t:.2f}s: {e}")
            continue
        pil_image = cgimage_to_pil(cg_image)
        if downscale and downscale > 1:
            pil_image = pil_image.resize(
                (pil_image.width // downscale, pil_image.height // downscale)
            )
        yield pil_image


def ocr_image(
    image: Image.Image, recognition_languages: Optional[List[str]] = None
) -> List[Tuple[str, Any]]:
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
        request.setRecognitionLanguages_(recognition_languages)
    handler.performRequests_error_([request], None)
    return results


def create_heatmap_image(
    heatmap: np.ndarray, contrast_boost: float = HEATMAP_DEFAULT_CONTRAST_BOOST
) -> Image.Image:
    """創建白底綠色熱區圖"""
    maxv = np.max(heatmap)
    if maxv == 0:
        maxv = 1

    if contrast_boost > 1.0:
        heatmap_enhanced = np.power(heatmap / maxv, 1 / contrast_boost) * maxv
    else:
        heatmap_enhanced = heatmap

    norm = (heatmap_enhanced / maxv * 255).astype(np.uint8)
    heat_img = Image.fromarray(norm, mode="L").convert("RGBA")

    r, g, b, a = heat_img.split()
    white_bg = Image.new("RGBA", heat_img.size, HEATMAP_WHITE_BG_COLOR)
    green_heat = Image.merge(
        "RGBA",
        (
            r.point(lambda v: 0),
            g.point(lambda v: min(255, int(v * HEATMAP_GREEN_MULTIPLIER))),
            b.point(lambda v: 0),
            a.point(lambda v: int(v * HEATMAP_ALPHA_MULTIPLIER)),
        ),
    )
    heat_img = Image.alpha_composite(white_bg, green_heat)

    return heat_img


def draw_grid_and_labels(
    img: Image.Image, grid_interval: int = HEATMAP_DEFAULT_GRID_INTERVAL
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype("Arial.ttf", HEATMAP_DEFAULT_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for x in range(0, width, grid_interval):
        draw.line([(x, 0), (x, height)], fill=HEATMAP_GRID_COLOR, width=1)
        draw.rectangle([(x + 2, 2), (x + 62, 32)], fill=HEATMAP_LABEL_BG_COLOR)
        draw.text((x + 5, 5), str(x), fill=HEATMAP_LABEL_TEXT_COLOR, font=font)

    for y in range(0, height, grid_interval):
        draw.line([(0, y), (width, y)], fill=HEATMAP_GRID_COLOR, width=1)
        draw.rectangle([(2, y + 2), (62, y + 32)], fill=HEATMAP_LABEL_BG_COLOR)
        draw.text((5, y + 5), str(y), fill=HEATMAP_LABEL_TEXT_COLOR, font=font)

    return img


def main(
    video_path: str,
    output_png: str,
    interval: float = DEFAULT_INTERVAL,
    downscale: int = DEFAULT_DOWNSCALE,
    recognition_languages: Optional[List[str]] = None,
    quiet: bool = False,
    grid_interval: int = HEATMAP_DEFAULT_GRID_INTERVAL,
    contrast_boost: float = HEATMAP_DEFAULT_CONTRAST_BOOST,
) -> None:
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    tracks = asset.tracks()
    original_width = None
    original_height = None
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
    if original_width is None or original_height is None:
        raise RuntimeError("Cannot determine video resolution.")

    duration = asset.duration().value / asset.duration().timescale
    total_frames = int(duration // interval)

    heatmap = np.zeros((original_height, original_width), dtype=np.uint32)

    frame_iter = extract_frames(video_path, interval, downscale, quiet=quiet)
    if HAS_TQDM:
        frame_iter = tqdm(frame_iter, total=total_frames, desc="Frames")
    for frame in frame_iter:
        frame_width, frame_height = frame.size
        ocr_results = ocr_image(frame, recognition_languages=recognition_languages)
        for text, bbox in ocr_results:
            if bbox is not None:
                x_norm = bbox.origin.x
                y_norm = bbox.origin.y
                w_norm = bbox.size.width
                h_norm = bbox.size.height
                x0_downscaled = x_norm * frame_width
                y0_downscaled = y_norm * frame_height
                x1_downscaled = (x_norm + w_norm) * frame_width
                y1_downscaled = (y_norm + h_norm) * frame_height
                x0 = int(x0_downscaled * downscale)
                y0 = int(y0_downscaled * downscale)
                x1 = int(x1_downscaled * downscale)
                y1 = int(y1_downscaled * downscale)
                y0_img = original_height - y1
                y1_img = original_height - y0
                x0 = max(0, min(original_width - 1, x0))
                x1 = max(0, min(original_width - 1, x1))
                y0_img = max(0, min(original_height - 1, y0_img))
                y1_img = max(0, min(original_height - 1, y1_img))
                heatmap[y0_img:y1_img, x0:x1] += 1

    heat_img = create_heatmap_image(heatmap, contrast_boost)
    heat_img = draw_grid_and_labels(heat_img, grid_interval=grid_interval)
    heat_img.save(output_png)
    logger.info(
        f"Heatmap saved to {output_png} (white background with green heatmap, contrast boost: {contrast_boost})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text heatmap from video using OCR."
    )
    parser.add_argument("input_video", help="Input video file")
    parser.add_argument("output_png", help="Output heatmap PNG file")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Frame interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=DEFAULT_DOWNSCALE,
        help="Downscale factor for frames before OCR (default: 2)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated OCR languages, e.g. ko,en",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress debug messages")
    parser.add_argument(
        "--grid-interval",
        type=int,
        default=HEATMAP_DEFAULT_GRID_INTERVAL,
        help="Grid interval in pixels for coordinate labels (default: 300)",
    )
    parser.add_argument(
        "--contrast-boost",
        type=float,
        default=HEATMAP_DEFAULT_CONTRAST_BOOST,
        help="Contrast boost factor (1.0=normal, 2.0=high contrast) (default: 1.0)",
    )
    args = parser.parse_args()
    recognition_languages = args.languages.split(",") if args.languages else None
    main(
        args.input_video,
        args.output_png,
        args.interval,
        args.downscale,
        recognition_languages,
        quiet=args.quiet,
        grid_interval=args.grid_interval,
        contrast_boost=args.contrast_boost,
    )
