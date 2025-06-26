import argparse
import os

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

# 嘗試匯入 tqdm
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


def extract_frames(video_path, interval=1.0, downscale=2, quiet=False):
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
                print(f"[DEBUG] Failed to extract frame at {t:.2f}s: {e}")
            continue
        pil_image = cgimage_to_pil(cg_image)
        if downscale and downscale > 1:
            pil_image = pil_image.resize(
                (pil_image.width // downscale, pil_image.height // downscale)
            )
        yield pil_image


def ocr_image(image, recognition_languages=None):
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
        request.setRecognitionLanguages_(recognition_languages)
    handler.performRequests_error_([request], None)
    return results


def create_heatmap_image(heatmap, contrast_boost=1.0):
    """
    創建白底綠色熱區圖
    """
    maxv = np.max(heatmap)
    if maxv == 0:
        maxv = 1

    # 應用對比度增強
    if contrast_boost > 1.0:
        # 使用冪次函數增強對比度
        heatmap_enhanced = np.power(heatmap / maxv, 1 / contrast_boost) * maxv
    else:
        heatmap_enhanced = heatmap

    norm = (heatmap_enhanced / maxv * 255).astype(np.uint8)
    heat_img = Image.fromarray(norm, mode="L").convert("RGBA")

    # 白底綠色熱區
    r, g, b, a = heat_img.split()
    # 創建白底
    white_bg = Image.new("RGBA", heat_img.size, (255, 255, 255, 255))
    # 綠色熱區
    green_heat = Image.merge(
        "RGBA",
        (
            r.point(lambda v: 0),
            g.point(lambda v: min(255, int(v * 1.5))),  # 加強綠色
            b.point(lambda v: 0),
            a.point(lambda v: int(v * 0.8)),
        ),
    )
    # 合成
    heat_img = Image.alpha_composite(white_bg, green_heat)

    return heat_img


def draw_grid_and_labels(img, grid_interval=300):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 白底上的格線和標籤顏色
    grid_color = (100, 100, 100, 200)  # 深灰色
    label_bg_color = (0, 0, 0, 180)  # 黑色背景
    label_text_color = (255, 255, 255, 255)  # 白色文字

    # 嘗試載入系統字體
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # 畫垂直格線與 x 座標
    for x in range(0, width, grid_interval):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        # 標註 x 座標
        draw.rectangle([(x + 2, 2), (x + 62, 32)], fill=label_bg_color)
        draw.text((x + 5, 5), str(x), fill=label_text_color, font=font)

    # 畫水平格線與 y 座標
    for y in range(0, height, grid_interval):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        # 標註 y 座標
        draw.rectangle([(2, y + 2), (62, y + 32)], fill=label_bg_color)
        draw.text((5, y + 5), str(y), fill=label_text_color, font=font)

    return img


def main(
    video_path,
    output_png,
    interval=1.0,
    downscale=2,
    recognition_languages=None,
    quiet=False,
    grid_interval=300,
    contrast_boost=1.0,
):
    # 取得原始分辨率
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

    # 計算總 frame 數
    duration = asset.duration().value / asset.duration().timescale
    total_frames = int(duration // interval)

    # 建立累積熱區圖
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
                # 轉成 downscaled 幀像素座標
                x0_downscaled = x_norm * frame_width
                y0_downscaled = y_norm * frame_height
                x1_downscaled = (x_norm + w_norm) * frame_width
                y1_downscaled = (y_norm + h_norm) * frame_height
                # 轉回原始分辨率
                x0 = int(x0_downscaled * downscale)
                y0 = int(y0_downscaled * downscale)
                x1 = int(x1_downscaled * downscale)
                y1 = int(y1_downscaled * downscale)
                # y 軸反轉（Vision 左下，ASS/圖像 左上）
                y0_img = original_height - y1
                y1_img = original_height - y0
                # 限制在圖像範圍內
                x0 = max(0, min(original_width - 1, x0))
                x1 = max(0, min(original_width - 1, x1))
                y0_img = max(0, min(original_height - 1, y0_img))
                y1_img = max(0, min(original_height - 1, y1_img))
                # 疊加熱區
                heatmap[y0_img:y1_img, x0:x1] += 1

    # 創建熱區圖
    heat_img = create_heatmap_image(heatmap, contrast_boost)

    # 加上格線與座標標註
    heat_img = draw_grid_and_labels(heat_img, grid_interval=grid_interval)
    heat_img.save(output_png)
    print(
        f"[INFO] Heatmap saved to {output_png} (white background with green heatmap, contrast boost: {contrast_boost})"
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
        default=1.0,
        help="Frame interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
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
        default=300,
        help="Grid interval in pixels for coordinate labels (default: 300)",
    )
    parser.add_argument(
        "--contrast-boost",
        type=float,
        default=1.0,
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
