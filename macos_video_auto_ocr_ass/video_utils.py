"""
影片處理工具模組

包含影片幀提取、OCR 處理等共用功能
"""

import io
import os
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import objc
from AVFoundation import AVAsset, AVAssetImageGenerator
from Cocoa import NSURL, NSData
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


def _safe_cgimage_extraction(generator, cm_time, quiet=False):
    """
    安全地從影片生成器中提取 CGImage

    Args:
        generator: AVAssetImageGenerator 實例
        cm_time: CMTime 時間點
        quiet: 是否安靜模式

    Returns:
        CGImage 或 None（如果提取失敗）
    """
    try:
        result = generator.copyCGImageAtTime_actualTime_error_(cm_time, None, None)
        if isinstance(result, tuple):
            cg_image = result[0]
        else:
            cg_image = result
        return cg_image
    except Exception as e:
        if not quiet:
            print(f"[DEBUG] Failed to extract frame: {e}")
        return None


def _process_frame_image(pil_image, downscale, scan_rect, quiet=False):
    """
    處理幀圖像：縮放和裁剪

    Args:
        pil_image: PIL 圖像
        downscale: 縮放因子
        scan_rect: 掃描區域
        quiet: 是否安靜模式

    Returns:
        (處理後的圖像, 裁剪偏移)
    """
    # 縮放處理
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

    if not quiet:
        print(
            f"[DEBUG] Processed frame, "
            f"PIL image size: {getattr(pil_image, 'size', None)}, "
            f"mode: {getattr(pil_image, 'mode', None)}, "
            f"crop_offset: {crop_offset}"
        )

    return pil_image, crop_offset


def cgimage_to_pil(cg_image) -> Image.Image:
    """將 CGImageRef 轉換為 PIL Image"""
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    buffer = bytes(data)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    arr = arr.reshape((height, width, 4))
    return Image.fromarray(arr[..., :3])


def get_video_info(video_path: str) -> Tuple[float, Optional[int], Optional[int]]:
    """獲取影片資訊：時長、寬度、高度"""
    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)

    # 獲取時長
    duration = asset.duration().value / asset.duration().timescale

    # 獲取分辨率
    original_width = None
    original_height = None
    tracks = asset.tracks()
    if tracks and len(tracks) > 0:
        for track in tracks:
            if track.mediaType() == "vide":
                natural_size = track.naturalSize()
                original_width = int(natural_size.width)
                original_height = int(natural_size.height)
                break

    return duration, original_width, original_height


def extract_frames(
    video_path: str,
    interval: float = 1.0,
    downscale: int = 2,
    quiet: bool = False,
    scan_rect: Optional[Tuple[int, int, int, int]] = None,
) -> Generator[Tuple[float, Image.Image, Tuple[int, int]], None, None]:
    """
    從影片中提取幀

    Args:
        video_path: 影片路徑
        interval: 提取間隔（秒）
        downscale: 縮放因子
        quiet: 是否安靜模式
        scan_rect: 掃描區域 (x, y, width, height)

    Yields:
        (時間戳, PIL圖像, 裁剪偏移)
    """
    if not quiet:
        print(f"[DEBUG] Entering extract_frames for {video_path}")

    url = NSURL.fileURLWithPath_(os.path.abspath(video_path))
    asset = AVAsset.assetWithURL_(url)
    generator = AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
    generator.setAppliesPreferredTrackTransform_(True)

    duration = asset.duration().value / asset.duration().timescale
    if not quiet:
        print(f"[DEBUG] Video duration: {duration} seconds")

    times = [CMTimeMakeWithSeconds(t, 600) for t in np.arange(0, duration, interval)]
    yielded = False

    for idx, (t, cm_time) in enumerate(zip(np.arange(0, duration, interval), times)):
        if t >= duration:
            if not quiet:
                print(
                    f"[DEBUG] Skipping frame at {t:.2f}s (beyond duration {duration:.2f}s)"
                )
            continue

        if not quiet:
            print(f"[DEBUG] Attempting to extract frame at {t:.2f}s (index {idx})")

        cg_image = _safe_cgimage_extraction(generator, cm_time, quiet)
        if cg_image is None:
            continue

        pil_image = cgimage_to_pil(cg_image)
        pil_image, crop_offset = _process_frame_image(
            pil_image, downscale, scan_rect, quiet
        )

        yielded = True
        yield t, pil_image, crop_offset

    if not yielded and not quiet:
        print("[DEBUG] No frames were yielded from extract_frames!")


def ocr_image(
    image: Image.Image,
    recognition_languages: Optional[List[str]] = None,
    quiet: bool = False,
) -> List[Tuple[str, object]]:
    """
    對圖像進行 OCR 識別

    Args:
        image: PIL 圖像
        recognition_languages: 識別語言列表
        quiet: 是否安靜模式

    Returns:
        OCR 結果列表，每個元素為 (文字, 邊界框)
    """
    if not quiet:
        print(
            f"[DEBUG] Entering ocr_image, image type: {type(image)}, size: {getattr(image, 'size', None)}"
        )

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
        if not quiet:
            print(f"[DEBUG] Setting recognition languages: {recognition_languages}")
        request.setRecognitionLanguages_(recognition_languages)

    handler.performRequests_error_([request], None)
    return results
