import asyncio
from face_helper import detect_face, detect_face_and_save, detect_from_bytes
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
import numpy as np
import cv2


app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)



app.mount("/static", StaticFiles(directory="static"), name="static")




@app.get("/")
async def root():
    return HTMLResponse(open("static/index.html").read())

# @app.post("/")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded file
#         contents = await file.read()
#         # Convert bytes to numpy array
#         nparr = np.frombuffer(contents, np.uint8)
#         # Decode image from numpy array
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return JSONResponse(status_code=400, content={"error": "Invalid image format"})
        
#         # Save the image with detected face
#         save_path = f"static/{file.filename}"
#         result=detect_face_and_save(image_path=img, save_path=save_path)
        
#         return JSONResponse(content={"emotion": result["emotion"], "image_path": save_path})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        # Predict emotion
        result = detect_from_bytes(data)
        print(f"Received data: {result["emotion"]}" if result else "No face detected")
        await websocket.send_json({"emotion": result["emotion"] if result else "No face detected",})
        