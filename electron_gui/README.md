# Electron GUI for macos-video-auto-ocr-ass

## 啟動方式

```bash
cd electron_gui
npm install
npm start
```

## 功能說明
- 目前提供簡單的影片檔案選擇與按鈕，預留與 Python 後端 API 溝通的介面。
- 未來可將 Python 程式（如 video_ocr_to_ass.py）包裝成 HTTP API，Electron 透過 HTTP 請求與之溝通。

## 串接 Python API 建議
1. 使用 [Flask](https://flask.palletsprojects.com/) 或 [FastAPI](https://fastapi.tiangolo.com/)，將 Python 主流程包裝成本機 HTTP 服務。
2. Electron 前端用 fetch/ajax 呼叫本機 API，取得 OCR/字幕結果。
3. 打包時可用 [PyInstaller](https://www.pyinstaller.org/) 或 [electron-builder](https://www.electron.build/) 將 Python 與 Electron 一起包進安裝檔。

---
如需範例 Python API 或整合細節，請再告知！ 