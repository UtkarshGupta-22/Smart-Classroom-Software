from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
from smart_classroom_core import api_register_student, api_start_classroom

app = FastAPI()

import glob

@app.get("/get-latest-attendance")
def get_latest_attendance():
    try:
        files = sorted(glob.glob("classroom_log_*.csv"), reverse=True)
        if not files:
            raise HTTPException(status_code=404, detail="No attendance file found.")
        df = pd.read_csv(files[0])
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class RegisterRequest(BaseModel):
    name: str


status = {"attendance_running": False}


@app.post("/register")
def register(req: RegisterRequest):
    try:
        threading.Thread(target=api_register_student, args=(req.name.strip().replace(" ", "_"),)).start()
        return {"status": f"Registration started for {req.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-session")
def start_session():
    if status["attendance_running"]:
        raise HTTPException(status_code=400, detail="Session already running.")

    def run():
        status["attendance_running"] = True
        api_start_classroom()
        status["attendance_running"] = False

    threading.Thread(target=run).start()
    return {"status": "Session started"}
