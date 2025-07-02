import sys

import objc
from Vision import VNRecognizeTextRequest

print(dir(VNRecognizeTextRequest))


def main():
    # 建立一個 instance
    request = VNRecognizeTextRequest.alloc().init()
    # 呼叫 instance method，傳入 None 當 error 參數
    result = request.supportedRecognitionLanguagesAndReturnError_(None)
    # result 可能是 tuple (languages, error)
    if isinstance(result, tuple):
        languages, error = result
    else:
        languages = result
        error = None
    if error is not None:
        print("取得語言時發生錯誤:", error)
        return
    print("支援的語言：")
    for lang in languages:
        print(lang)


if __name__ == "__main__":
    main()
