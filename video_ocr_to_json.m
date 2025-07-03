#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>

void printUsage(const char *programName) {
    NSLog(@"Usage: %s <video_path> <output_json> [interval] [downscale] [scan_rect_x,scan_rect_y,scan_rect_w,scan_rect_h] [languages] [start_time] [end_time] [show_progress]", programName);
    NSLog(@"  - interval: Time interval between frames in seconds (default: 1.0)");
    NSLog(@"  - downscale: Downscale factor for frames (default: 1)");
    NSLog(@"  - scan_rect: Scan region in format x,y,width,height (pixel coordinates, default: entire frame)");
    NSLog(@"  - languages: Comma-separated language codes (e.g., ja,en,zh-Hant)");
    NSLog(@"  - start_time: Start time in seconds (default: 0.0)");
    NSLog(@"  - end_time: End time in seconds (default: video duration)");
    NSLog(@"  - show_progress: 1 to show progress, 0 to hide (default: 0)");
}

CGImageRef createDownscaledImage(CGImageRef originalImage, int downscale) {
    size_t originalWidth = CGImageGetWidth(originalImage);
    size_t originalHeight = CGImageGetHeight(originalImage);
    size_t newWidth = originalWidth / downscale;
    size_t newHeight = originalHeight / downscale;
    
    // 建立目標 context
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(NULL,
                                               newWidth,
                                               newHeight,
                                               8,
                                               newWidth * 4,
                                               colorSpace,
                                               kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(colorSpace);
    
    // 繪製縮小的圖像
    CGContextDrawImage(context,
                      CGRectMake(0, 0, newWidth, newHeight),
                      originalImage);
    
    // 取得結果
    CGImageRef newImage = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    
    return newImage;
}

CGImageRef cropImage(CGImageRef originalImage, CGRect cropRect) {
    CGImageRef croppedImage = CGImageCreateWithImageInRect(originalImage, cropRect);
    return croppedImage;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }
        
        // 解析參數
        NSString *videoPath = @(argv[1]);
        NSString *outputPath = @(argv[2]);
        float interval = (argc > 3) ? atof(argv[3]) : 1.0;
        int downscale = (argc > 4) ? atoi(argv[4]) : 1;
        
        // 解析掃描區域（像素座標）
        CGRect scanRect = CGRectNull;
        if (argc > 5) {
            NSString *scanRectStr = @(argv[5]);
            NSArray *components = [scanRectStr componentsSeparatedByString:@","];
            if (components.count == 4) {
                scanRect = CGRectMake([components[0] floatValue],
                                    [components[1] floatValue],
                                    [components[2] floatValue],
                                    [components[3] floatValue]);
            }
        }
        
        // 解析語言設定
        NSArray *languages = nil;
        if (argc > 6) {
            NSString *languagesStr = @(argv[6]);
            if ([languagesStr isEqualToString:@"__AUTO__"] || [languagesStr length] == 0) {
                languages = nil;
            } else {
                languages = [languagesStr componentsSeparatedByString:@","];
            }
        }
        float start_time = (argc > 7) ? atof(argv[7]) : 0.0;
        float end_time = (argc > 8) ? atof(argv[8]) : -1.0;
        int show_progress = (argc > 9) ? atoi(argv[9]) : 0;
        
        NSURL *url = [NSURL fileURLWithPath:videoPath];
        AVAsset *asset = [AVAsset assetWithURL:url];
        CMTime duration = asset.duration;
        float durationSeconds = CMTimeGetSeconds(duration);
        NSLog(@"[DEBUG] durationSeconds: %.3f, start_time: %.3f, end_time: %.3f", durationSeconds, start_time, end_time);
        
        if (end_time < 0.0 || end_time > durationSeconds) {
            end_time = durationSeconds;
        }
        if (start_time < 0.0) {
            start_time = 0.0;
        }
        if (start_time >= end_time) {
            NSLog(@"start_time must be less than end_time");
            return 1;
        }
        
        int total_frames = (int)((end_time - start_time) / interval);
        int progress_step = total_frames / 10; // 每 10% 顯示一次
        if (progress_step < 1) progress_step = 1;
        int frame_count = 0;
        
        AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
        generator.appliesPreferredTrackTransform = YES;
        
        NSMutableArray *resultsArray = [NSMutableArray array];
        
        for (float t = start_time; t < end_time; t += interval) {
            @autoreleasepool {
                CMTime time = CMTimeMakeWithSeconds(t, 600);
                NSError *error = nil;
                CGImageRef cgImage = [generator copyCGImageAtTime:time actualTime:nil error:&error];
                if (error || !cgImage) {
                    NSLog(@"Failed to extract frame at %.2fs: %@", t, error);
                    continue;
                }
                
                // 如果需要，進行裁剪
                CGImageRef processedImage = cgImage;
                if (!CGRectIsNull(scanRect)) {
                    processedImage = cropImage(cgImage, scanRect);
                    CGImageRelease(cgImage);
                    cgImage = processedImage;
                }
                
                // 如果需要，進行縮放
                if (downscale > 1) {
                    processedImage = createDownscaledImage(cgImage, downscale);
                    if (cgImage != [generator copyCGImageAtTime:time actualTime:nil error:&error]) {
                        CGImageRelease(cgImage);
                    }
                    cgImage = processedImage;
                }
                
                VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:nil];
                request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
                request.automaticallyDetectsLanguage = (languages == nil);
                
                // 設定語言（如果指定了）
                if (languages) {
                    request.recognitionLanguages = languages;
                }
                
                VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
                NSError *ocrError = nil;
                [handler performRequests:@[request] error:&ocrError];
                
                NSMutableArray *frameResults = [NSMutableArray array];
                if (!ocrError) {
                    for (VNRecognizedTextObservation *obs in request.results) {
                        VNRecognizedText *top = [[obs topCandidates:1] firstObject];
                        if (top) {
                            CGRect bbox = obs.boundingBox;
                            // 如果使用了掃描區域，需要調整座標回到原始影片座標系統
                            if (!CGRectIsNull(scanRect)) {
                                bbox = CGRectMake(
                                    scanRect.origin.x + bbox.origin.x * scanRect.size.width,
                                    scanRect.origin.y + bbox.origin.y * scanRect.size.height,
                                    bbox.size.width * scanRect.size.width,
                                    bbox.size.height * scanRect.size.height
                                );
                            }
                            NSDictionary *item = @{
                                @"text": top.string ?: @"",
                                @"bbox": @[@(bbox.origin.x), @(bbox.origin.y), @(bbox.size.width), @(bbox.size.height)]
                            };
                            [frameResults addObject:item];
                        }
                    }
                }
                
                NSDictionary *frameDict = @{
                    @"timestamp": @(t),
                    @"results": frameResults
                };
                [resultsArray addObject:frameDict];
                
                CGImageRelease(cgImage);
                frame_count++;
                if (show_progress && (frame_count % progress_step == 0 || frame_count == total_frames)) {
                    int percent = (int)((float)frame_count / total_frames * 100);
                    NSLog(@"Progress: %d/%d (約 %d%%)", frame_count, total_frames, percent);
                }
            }
        }
        
        // 輸出 JSON
        NSError *jsonError = nil;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:resultsArray options:NSJSONWritingPrettyPrinted error:&jsonError];
        if (!jsonData || jsonError) {
            NSLog(@"Failed to serialize JSON: %@", jsonError);
            return 1;
        }
        if (![jsonData writeToFile:outputPath atomically:YES]) {
            NSLog(@"Failed to write JSON to %@", outputPath);
            return 1;
        }
        NSLog(@"Done! OCR results saved to %@", outputPath);
    }
    return 0;
} 