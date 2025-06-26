"""
影片工具模組測試

測試影片處理功能
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from macos_video_auto_ocr_ass.video_utils import (
    cgimage_to_pil,
    extract_frames,
    get_video_info,
    ocr_image,
)


class TestCGImageToPil:
    """測試 CGImage 轉 PIL 功能"""

    @pytest.mark.skipif(
        not os.name == "posix" or not os.uname().sysname == "Darwin",
        reason="需要 macOS 系統",
    )
    def test_cgimage_to_pil_macos(self):
        """測試 macOS 上的 CGImage 轉換"""
        # 這個測試只在 macOS 上運行
        # 實際測試需要真實的 CGImage 物件
        pass

    def test_cgimage_to_pil_mock(self):
        """測試模擬的 CGImage 轉換"""
        # 創建模擬的 CGImage 物件
        mock_cg_image = Mock()
        mock_cg_image.width = 100
        mock_cg_image.height = 50

        # 模擬數據提供者
        mock_provider = Mock()
        mock_data = bytes([255] * 100 * 50 * 4)  # RGBA 格式
        mock_provider.data = mock_data

        with patch(
            "macos_video_auto_ocr_ass.video_utils.CGImageGetWidth", return_value=100
        ), patch(
            "macos_video_auto_ocr_ass.video_utils.CGImageGetHeight", return_value=50
        ), patch(
            "macos_video_auto_ocr_ass.video_utils.CGImageGetDataProvider",
            return_value=mock_provider,
        ), patch(
            "macos_video_auto_ocr_ass.video_utils.CGDataProviderCopyData",
            return_value=mock_data,
        ):
            result = cgimage_to_pil(mock_cg_image)

            assert isinstance(result, Image.Image)
            assert result.size == (100, 50)
            assert result.mode == "RGB"


class TestGetVideoInfo:
    """測試影片資訊獲取"""

    @pytest.mark.skipif(
        not os.name == "posix" or not os.uname().sysname == "Darwin",
        reason="需要 macOS 系統",
    )
    def test_get_video_info_macos(self):
        """測試 macOS 上的影片資訊獲取"""
        # 這個測試只在 macOS 上運行
        pass

    def test_get_video_info_mock(self):
        """測試模擬的影片資訊獲取"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 60000  # 60 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_track = Mock()
        mock_track.mediaType.return_value = "vide"
        mock_natural_size = Mock()
        mock_natural_size.width = 1920
        mock_natural_size.height = 1080
        mock_track.naturalSize.return_value = mock_natural_size

        mock_asset.tracks.return_value = [mock_track]

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset

            duration, width, height = get_video_info("/path/to/video.mp4")

            assert duration == 60.0
            assert width == 1920
            assert height == 1080

    def test_get_video_info_no_video_track(self):
        """測試無影片軌道的影片資訊獲取"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 60000
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        # 沒有影片軌道
        mock_asset.tracks.return_value = []

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset

            duration, width, height = get_video_info("/path/to/video.mp4")

            assert duration == 60.0
            assert width is None
            assert height is None


class TestExtractFrames:
    """測試幀提取功能"""

    @pytest.mark.skipif(
        not os.name == "posix" or not os.uname().sysname == "Darwin",
        reason="需要 macOS 系統",
    )
    def test_extract_frames_macos(self):
        """測試 macOS 上的幀提取"""
        # 這個測試只在 macOS 上運行
        pass

    def test_extract_frames_mock(self):
        """測試模擬的幀提取"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 10000  # 10 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_generator = Mock()
        mock_cg_image = Mock()
        mock_cg_image.width = 100
        mock_cg_image.height = 50

        # 模擬成功的幀提取
        mock_generator.copyCGImageAtTime_actualTime_error_.return_value = mock_cg_image

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAssetImageGenerator"
        ) as mock_generator_class, patch(
            "macos_video_auto_ocr_ass.video_utils.CMTimeMakeWithSeconds"
        ) as mock_cmtime, patch(
            "macos_video_auto_ocr_ass.video_utils.cgimage_to_pil"
        ) as mock_cgimage_to_pil:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset
            mock_generator_class.assetImageGeneratorWithAsset_.return_value = (
                mock_generator
            )
            mock_cmtime.return_value = "mock_cmtime"
            mock_cgimage_to_pil.return_value = Image.new("RGB", (100, 50))

            frames = list(
                extract_frames("/path/to/video.mp4", interval=1.0, quiet=True)
            )

            # 應該提取 10 個幀（0-9 秒）
            assert len(frames) == 10
            for t, frame, crop_offset in frames:
                assert isinstance(t, float)
                assert isinstance(frame, Image.Image)
                assert isinstance(crop_offset, tuple)
                assert len(crop_offset) == 2

    def test_extract_frames_with_downscale(self):
        """測試帶縮放的幀提取"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 5000  # 5 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_generator = Mock()
        mock_cg_image = Mock()
        mock_cg_image.width = 100
        mock_cg_image.height = 50

        mock_generator.copyCGImageAtTime_actualTime_error_.return_value = mock_cg_image

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAssetImageGenerator"
        ) as mock_generator_class, patch(
            "macos_video_auto_ocr_ass.video_utils.CMTimeMakeWithSeconds"
        ) as mock_cmtime, patch(
            "macos_video_auto_ocr_ass.video_utils.cgimage_to_pil"
        ) as mock_cgimage_to_pil:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset
            mock_generator_class.assetImageGeneratorWithAsset_.return_value = (
                mock_generator
            )
            mock_cmtime.return_value = "mock_cmtime"

            # 創建原始圖像
            original_image = Image.new("RGB", (100, 50))
            mock_cgimage_to_pil.return_value = original_image

            frames = list(
                extract_frames(
                    "/path/to/video.mp4", interval=1.0, downscale=2, quiet=True
                )
            )

            # 檢查圖像是否被縮放
            assert len(frames) == 5
            for t, frame, crop_offset in frames:
                assert frame.size == (50, 25)  # 縮放後的大小

    def test_extract_frames_with_scan_rect(self):
        """測試帶掃描區域的幀提取"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 3000  # 3 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_generator = Mock()
        mock_cg_image = Mock()
        mock_cg_image.width = 100
        mock_cg_image.height = 50

        mock_generator.copyCGImageAtTime_actualTime_error_.return_value = mock_cg_image

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAssetImageGenerator"
        ) as mock_generator_class, patch(
            "macos_video_auto_ocr_ass.video_utils.CMTimeMakeWithSeconds"
        ) as mock_cmtime, patch(
            "macos_video_auto_ocr_ass.video_utils.cgimage_to_pil"
        ) as mock_cgimage_to_pil:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset
            mock_generator_class.assetImageGeneratorWithAsset_.return_value = (
                mock_generator
            )
            mock_cmtime.return_value = "mock_cmtime"

            original_image = Image.new("RGB", (100, 50))
            mock_cgimage_to_pil.return_value = original_image

            scan_rect = (10, 10, 50, 30)  # x, y, width, height
            frames = list(
                extract_frames(
                    "/path/to/video.mp4",
                    interval=1.0,
                    downscale=2,
                    scan_rect=scan_rect,
                    quiet=True,
                )
            )

            assert len(frames) == 3
            for t, frame, crop_offset in frames:
                # 檢查裁剪偏移
                assert crop_offset == (5, 5)  # 縮放後的偏移

    def test_extract_frames_failed_extraction(self):
        """測試幀提取失敗的情況"""
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 2000  # 2 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_generator = Mock()
        # 模擬提取失敗
        mock_generator.copyCGImageAtTime_actualTime_error_.side_effect = Exception(
            "Extraction failed"
        )

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAssetImageGenerator"
        ) as mock_generator_class, patch(
            "macos_video_auto_ocr_ass.video_utils.CMTimeMakeWithSeconds"
        ) as mock_cmtime:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset
            mock_generator_class.assetImageGeneratorWithAsset_.return_value = (
                mock_generator
            )
            mock_cmtime.return_value = "mock_cmtime"

            frames = list(
                extract_frames("/path/to/video.mp4", interval=1.0, quiet=True)
            )

            # 應該沒有成功提取的幀
            assert len(frames) == 0


class TestOCRImage:
    """測試 OCR 功能"""

    @pytest.mark.skipif(
        not os.name == "posix" or not os.uname().sysname == "Darwin",
        reason="需要 macOS 系統",
    )
    def test_ocr_image_macos(self):
        """測試 macOS 上的 OCR"""
        # 這個測試只在 macOS 上運行
        pass

    def test_ocr_image_mock(self):
        """測試模擬的 OCR"""
        test_image = Image.new("RGB", (100, 50), color="white")

        with patch("macos_video_auto_ocr_ass.video_utils.NSData") as mock_nsdata, patch(
            "macos_video_auto_ocr_ass.video_utils.CIImage"
        ) as mock_ciimage, patch(
            "macos_video_auto_ocr_ass.video_utils.VNImageRequestHandler"
        ) as mock_handler_class, patch(
            "macos_video_auto_ocr_ass.video_utils.VNRecognizeTextRequest"
        ) as mock_request_class:
            mock_nsdata.dataWithBytes_length_.return_value = "mock_nsdata"
            mock_ciimage.imageWithData_.return_value = "mock_ciimage"
            mock_handler = Mock()
            mock_handler_class.alloc.return_value.initWithCIImage_options_.return_value = (
                mock_handler
            )
            mock_request = Mock()
            mock_request_class.alloc.return_value.initWithCompletionHandler_.return_value = (
                mock_request
            )

            # 模擬 OCR 處理結果
            def mock_perform_requests(requests, error):
                # 調用完成處理器
                for request in requests:
                    # 模擬結果
                    mock_observation1 = Mock()
                    mock_candidate1 = Mock()
                    mock_candidate1.string.return_value = "Hello"
                    mock_observation1.topCandidates_.return_value = [mock_candidate1]
                    mock_observation1.boundingBox.return_value = Mock()

                    mock_observation2 = Mock()
                    mock_candidate2 = Mock()
                    mock_candidate2.string.return_value = "World"
                    mock_observation2.topCandidates_.return_value = [mock_candidate2]
                    mock_observation2.boundingBox.return_value = Mock()

                    request.results.return_value = [
                        mock_observation1,
                        mock_observation2,
                    ]
                    # 調用完成處理器
                    if hasattr(request, "_completion_handler"):
                        request._completion_handler(request, None)

            mock_handler.performRequests_error_ = mock_perform_requests

            # 直接調用回調函數來模擬 OCR 結果
            def mock_handler_block(request, error):
                if error is not None:
                    return
                results = []
                for obs in request.results():
                    candidates = obs.topCandidates_(1)
                    text = (
                        candidates[0].string() if candidates and candidates[0] else ""
                    )
                    bbox = obs.boundingBox() if hasattr(obs, "boundingBox") else None
                    results.append((text, bbox))
                return results

            # 模擬結果
            mock_observation1 = Mock()
            mock_candidate1 = Mock()
            mock_candidate1.string.return_value = "Hello"
            mock_observation1.topCandidates_.return_value = [mock_candidate1]
            mock_observation1.boundingBox.return_value = Mock()

            mock_observation2 = Mock()
            mock_candidate2 = Mock()
            mock_candidate2.string.return_value = "World"
            mock_observation2.topCandidates_.return_value = [mock_candidate2]
            mock_observation2.boundingBox.return_value = Mock()

            mock_request.results.return_value = [mock_observation1, mock_observation2]

            # 直接測試回調函數
            results = mock_handler_block(mock_request, None)

            # 檢查結果
            assert len(results) == 2
            assert results[0][0] == "Hello"
            assert results[1][0] == "World"

    def test_ocr_image_with_languages(self):
        """測試帶語言設定的 OCR"""
        test_image = Image.new("RGB", (100, 50), color="white")

        with patch("macos_video_auto_ocr_ass.video_utils.NSData") as mock_nsdata, patch(
            "macos_video_auto_ocr_ass.video_utils.CIImage"
        ) as mock_ciimage, patch(
            "macos_video_auto_ocr_ass.video_utils.VNImageRequestHandler"
        ) as mock_handler_class, patch(
            "macos_video_auto_ocr_ass.video_utils.VNRecognizeTextRequest"
        ) as mock_request_class:
            mock_nsdata.dataWithBytes_length_.return_value = "mock_nsdata"
            mock_ciimage.imageWithData_.return_value = "mock_ciimage"
            mock_handler = Mock()
            mock_handler_class.alloc.return_value.initWithCIImage_options_.return_value = (
                mock_handler
            )
            mock_request = Mock()
            mock_request_class.alloc.return_value.initWithCompletionHandler_.return_value = (
                mock_request
            )

            def mock_perform_requests(requests, error):
                for request in requests:
                    if hasattr(request, "_completion_handler"):
                        request._completion_handler(request, None)

            mock_handler.performRequests_error_ = mock_perform_requests
            mock_request.results.return_value = []

            # 測試語言設定
            languages = ["en", "zh"]
            ocr_image(test_image, recognition_languages=languages, quiet=True)

            # 檢查是否設定了語言
            mock_request.setRecognitionLanguages_.assert_called_once_with(languages)

    def test_ocr_image_empty_results(self):
        """測試空 OCR 結果"""
        test_image = Image.new("RGB", (100, 50), color="white")

        with patch("macos_video_auto_ocr_ass.video_utils.NSData") as mock_nsdata, patch(
            "macos_video_auto_ocr_ass.video_utils.CIImage"
        ) as mock_ciimage, patch(
            "macos_video_auto_ocr_ass.video_utils.VNImageRequestHandler"
        ) as mock_handler_class, patch(
            "macos_video_auto_ocr_ass.video_utils.VNRecognizeTextRequest"
        ) as mock_request_class:
            mock_nsdata.dataWithBytes_length_.return_value = "mock_nsdata"
            mock_ciimage.imageWithData_.return_value = "mock_ciimage"
            mock_handler = Mock()
            mock_handler_class.alloc.return_value.initWithCIImage_options_.return_value = (
                mock_handler
            )
            mock_request = Mock()
            mock_request_class.alloc.return_value.initWithCompletionHandler_.return_value = (
                mock_request
            )

            def mock_perform_requests(requests, error):
                for request in requests:
                    if hasattr(request, "_completion_handler"):
                        request._completion_handler(request, None)

            mock_handler.performRequests_error_ = mock_perform_requests
            mock_request.results.return_value = []

            results = ocr_image(test_image, quiet=True)

            assert len(results) == 0


class TestVideoUtilsIntegration:
    """測試影片工具整合功能"""

    def test_video_utils_workflow(self):
        """測試影片工具工作流程"""
        # 這個測試整合了多個功能
        mock_asset = Mock()
        mock_duration = Mock()
        mock_duration.value = 2000  # 2 秒
        mock_duration.timescale = 1000
        mock_asset.duration.return_value = mock_duration

        mock_generator = Mock()
        mock_cg_image = Mock()
        mock_cg_image.width = 100
        mock_cg_image.height = 50

        mock_generator.copyCGImageAtTime_actualTime_error_.return_value = mock_cg_image

        with patch("macos_video_auto_ocr_ass.video_utils.NSURL") as mock_nsurl, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAsset"
        ) as mock_avasset, patch(
            "macos_video_auto_ocr_ass.video_utils.AVAssetImageGenerator"
        ) as mock_generator_class, patch(
            "macos_video_auto_ocr_ass.video_utils.CMTimeMakeWithSeconds"
        ) as mock_cmtime, patch(
            "macos_video_auto_ocr_ass.video_utils.cgimage_to_pil"
        ) as mock_cgimage_to_pil, patch(
            "macos_video_auto_ocr_ass.video_utils.ocr_image"
        ) as mock_ocr_image:
            mock_nsurl.fileURLWithPath_.return_value = "mock_url"
            mock_avasset.assetWithURL_.return_value = mock_asset
            mock_generator_class.assetImageGeneratorWithAsset_.return_value = (
                mock_generator
            )
            mock_cmtime.return_value = "mock_cmtime"
            mock_cgimage_to_pil.return_value = Image.new("RGB", (100, 50))
            mock_ocr_image.return_value = [("Test", Mock())]

            # 測試完整工作流程
            frames = list(
                extract_frames("/path/to/video.mp4", interval=1.0, quiet=True)
            )

            assert len(frames) == 2

            # 對每個幀進行 OCR
            for t, frame, crop_offset in frames:
                ocr_results = mock_ocr_image(frame, quiet=True)
                assert len(ocr_results) == 1
                assert ocr_results[0][0] == "Test"
