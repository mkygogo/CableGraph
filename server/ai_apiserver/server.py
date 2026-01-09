import uvicorn
from uvicorn.server import logger
from fastapi import FastAPI, File, UploadFile, HTTPException
from uuid import uuid4
import os
import shutil
import asyncio
from enum import Enum
from pathlib import Path
import yolo_detect
import json


app = FastAPI()  
  
# 确保 "files" 文件夹存在
os.makedirs("files", exist_ok=True)

# 定义任务状态枚举
class TaskStatus(str, Enum):
    NO_TASK = "no_task"
    WORKING = "working"
    FINISHED = "finished"

# 定义任务类
class Task:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task_status = TaskStatus.NO_TASK
        self.detection_ret = None

# 任务字典，用于存储任务状态
tasks = {}

# async def detect_img(task: Task, file_path: str):
#     task.task_status = TaskStatus.WORKING
#     # 模拟一个耗时的图片处理过程
#     ret  = yolo_detect.yolo_img_detect(file_path)

#     task.task_status = TaskStatus.FINISHED
#     task.detection_ret = ret
async def detect_img(task: Task, file_path: str):
    task.task_status = TaskStatus.WORKING
    try:
        # 调用升级后的接口
        ret = yolo_detect.cable_detect(file_path)
        task.detection_ret = ret
        task.task_status = TaskStatus.FINISHED
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        task.task_status = TaskStatus.FINISHED
        task.detection_ret = {"error": str(e)}


JSON_PATH = Path("D:/ai/cv/ocr/__temp__/results.json")
@app.get("/VoltageCurrentValue")  
async def getVoltageCurrentValue():  
    """返回JSON文件内容的接口"""
    try:
        # 读取并解析JSON文件
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "JSON文件未找到"}
    except json.JSONDecodeError:
        return {"error": "JSON解析失败"}
    except Exception as e:
        return {"error": f"服务器错误: {str(e)}"} 

@app.get("/")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    task_id = str(uuid4())
    task = Task(task_id)
    tasks[task_id] = task

    file_extension = os.path.splitext(file.filename)[1]
    file_name = f"{task_id}{file_extension}"
    file_path = os.path.join("files", file_name)

    with open(file_path, "wb") as buffer:
    # 使用 UploadFile 的 file 属性来访问上传的文件
        shutil.copyfileobj(file.file, buffer)

    # 异步调用图片处理函数
    asyncio.create_task(detect_img(task, file_path))

    return {"taskid": task_id}

@app.get("/check_task/{task_id}")
def check_task(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    response = {"task_status": task.task_status}
    if task.task_status == TaskStatus.FINISHED and task.detection_ret is not None:
        # 因为 cable_detect 现在返回的是普通的 dict，不需要再调用 to_dict()
        response["results"] = task.detection_ret
   
    return response
# def check_task(task_id: str):
#     # 检查任务是否存在
#     task = tasks.get(task_id)
#     if task is None:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     response = {"task_status": task.task_status}
#     if task.task_status == TaskStatus.FINISHED and task.detection_ret is not None:
#         detection_dicts = [detection.to_dict() for detection in task.detection_ret]
#         response["detection_ret"] = detection_dicts
   
#     return response

if __name__ == "__main__":
    #uvicorn.run(host="0.0.0.0", port=8920, app=app, debug=True)
    uvicorn.run(host="0.0.0.0", port=8920, app=app)