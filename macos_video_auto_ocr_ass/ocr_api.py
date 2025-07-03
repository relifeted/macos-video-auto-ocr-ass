import os
import subprocess
import tempfile
import threading
import uuid
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

ocr_tasks = {}
ocr_progress = {}


class OcrRequest(BaseModel):
    video_path: str
    interval: float = 1.0
    languages: str = None
    downscale: int = 2
    base_font_size: int = 24
    quiet: bool = True


@app.post("/ocr_ass")
async def ocr_ass(req: OcrRequest):
    video_path = req.video_path
    interval = req.interval
    languages = req.languages
    downscale = req.downscale
    base_font_size = req.base_font_size
    quiet = req.quiet

    if not video_path or not os.path.exists(video_path):
        return JSONResponse(content={"error": "video_path 不存在"}, status_code=400)

    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        output_ass = tmp.name

    cmd = [
        "python3",
        "video_ocr_to_ass.py",
        video_path,
        output_ass,
        "--interval",
        str(interval),
        "--downscale",
        str(downscale),
        "--base-font-size",
        str(base_font_size),
        "--quiet",
    ]
    if languages:
        cmd += ["--languages", languages]

    try:
        subprocess.run(cmd, check=True)
        with open(output_ass, "r", encoding="utf-8") as f:
            ass_content = f.read()
        os.unlink(output_ass)
        return {"ass": ass_content}
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class OcrObjcRequest(BaseModel):
    video_path: str
    interval: float = 1.0
    languages: str = None
    downscale: int = 1
    base_font_size: int = 24
    quiet: bool = True


@app.post("/start_ocr_ass_objc")
async def start_ocr_ass_objc(req: OcrObjcRequest):
    task_id = str(uuid.uuid4())
    ocr_progress[task_id] = {
        "status": "pending",
        "progress": 0,
        "ass": None,
        "error": None,
    }

    def run_ocr():
        try:
            video_path = req.video_path
            interval = req.interval
            languages = req.languages
            downscale = req.downscale
            base_font_size = req.base_font_size
            quiet = req.quiet
            if not video_path or not os.path.exists(video_path):
                ocr_progress[task_id]["status"] = "error"
                ocr_progress[task_id]["error"] = "video_path 不存在"
                return
            with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
                output_ass = tmp.name
            cmd = [
                "python3",
                "video_ocr_to_ass_objc.py",
                video_path,
                output_ass,
                "--interval",
                str(interval),
                "--downscale",
                str(downscale),
                "--base-font-size",
                str(base_font_size),
                "--quiet",
            ]
            if languages:
                cmd += ["--languages", languages]
            # 進度模擬：實際應改為 video_ocr_to_ass_objc.py 支援進度回報
            ocr_progress[task_id]["status"] = "running"
            proc = subprocess.Popen(cmd)
            while proc.poll() is None:
                # 這裡可根據實際情況更新進度
                ocr_progress[task_id]["progress"] += 5
                if ocr_progress[task_id]["progress"] > 95:
                    ocr_progress[task_id]["progress"] = 95
                import time

                time.sleep(1)
            if proc.returncode == 0:
                with open(output_ass, "r", encoding="utf-8") as f:
                    ass_content = f.read()
                os.unlink(output_ass)
                ocr_progress[task_id]["status"] = "done"
                ocr_progress[task_id]["progress"] = 100
                ocr_progress[task_id]["ass"] = ass_content
            else:
                ocr_progress[task_id]["status"] = "error"
                ocr_progress[task_id][
                    "error"
                ] = f"Process failed, code {proc.returncode}"
        except Exception as e:
            ocr_progress[task_id]["status"] = "error"
            ocr_progress[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_ocr)
    thread.start()
    ocr_tasks[task_id] = thread
    return {"task_id": task_id}


@app.get("/ocr_progress")
async def get_ocr_progress(task_id: str):
    if task_id not in ocr_progress:
        return JSONResponse(content={"error": "無此任務"}, status_code=404)
    return ocr_progress[task_id]


class MergeAssRequest(BaseModel):
    ass_content: Optional[str] = None
    ass_path: Optional[str] = None
    position_tolerance: int = 10
    time_gap_threshold: int = 500
    base_font_size: int = 24


@app.post("/merge_ass")
async def merge_ass(req: MergeAssRequest):
    if not req.ass_content and not req.ass_path:
        return JSONResponse(
            content={"error": "請提供 ass_content 或 ass_path"}, status_code=400
        )
    # 儲存臨時檔案
    if req.ass_content:
        with tempfile.NamedTemporaryFile(
            suffix=".ass", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            input_ass = tmp.name
            tmp.write(req.ass_content)
    else:
        input_ass = req.ass_path
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        output_ass = tmp.name
    cmd = [
        "python3",
        "merge_ass_subs.py",
        input_ass,
        output_ass,
        "--position-tolerance",
        str(req.position_tolerance),
        "--time-gap-threshold",
        str(req.time_gap_threshold),
        "--base-font-size",
        str(req.base_font_size),
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(output_ass, "r", encoding="utf-8") as f:
            merged_content = f.read()
        os.unlink(output_ass)
        if req.ass_content:
            os.unlink(input_ass)
        return {"ass": merged_content}
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class TranslateAssRequest(BaseModel):
    ass_content: Optional[str] = None
    ass_path: Optional[str] = None
    src_lang: str = "en"
    tgt_lang: str = "zh"
    model: Optional[str] = None
    device: str = "cpu"
    show_text: bool = False


@app.post("/translate_ass")
async def translate_ass(req: TranslateAssRequest):
    if not req.ass_content and not req.ass_path:
        return JSONResponse(
            content={"error": "請提供 ass_content 或 ass_path"}, status_code=400
        )
    # 儲存臨時檔案
    if req.ass_content:
        with tempfile.NamedTemporaryFile(
            suffix=".ass", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            input_ass = tmp.name
            tmp.write(req.ass_content)
    else:
        input_ass = req.ass_path
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        output_ass = tmp.name
    cmd = [
        "python3",
        "translate_ass_marianmt.py",
        input_ass,
        output_ass,
        "--src-lang",
        req.src_lang,
        "--tgt-lang",
        req.tgt_lang,
        "--device",
        req.device,
    ]
    if req.model:
        cmd += ["--model", req.model]
    if req.show_text:
        cmd += ["--show-text"]
    try:
        subprocess.run(cmd, check=True)
        with open(output_ass, "r", encoding="utf-8") as f:
            translated_content = f.read()
        os.unlink(output_ass)
        if req.ass_content:
            os.unlink(input_ass)
        return {"ass": translated_content}
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class TranslateLineRequest(BaseModel):
    text: str
    src_lang: str = "en"
    tgt_lang: str = "zh"
    model: str = None
    device: str = "cpu"


@app.post("/translate_line")
async def translate_line(req: TranslateLineRequest):
    try:
        # 預設 MarianMT 模型自動推斷
        if req.model:
            model_name = req.model
        else:
            # 這裡可根據 src_lang, tgt_lang 自動推斷模型名稱
            model_name = f"Helsinki-NLP/opus-mt-{req.src_lang}-{req.tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(
            **tokenizer([req.text], return_tensors="pt", padding=True)
        )
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return {"translated": tgt_text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    return {"video_path": tmp_path}


if __name__ == "__main__":
    uvicorn.run(
        "macos_video_auto_ocr_ass.ocr_api:app", host="0.0.0.0", port=5001, reload=True
    )
