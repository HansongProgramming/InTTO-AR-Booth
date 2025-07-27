# main.py (FastAPI AR booth app)

import cv2
import numpy as np
from cv2 import aruco
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
import time
import threading

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Marker ID to image
marker_images = {
    0: cv2.imread("static/images/1.png", cv2.IMREAD_UNCHANGED),
    1: cv2.imread("static/images/2.png", cv2.IMREAD_UNCHANGED),
    2: cv2.imread("static/images/3.png", cv2.IMREAD_UNCHANGED)
}

# Globals
frame_lock = threading.Lock()
latest_frame = None
os.makedirs("photos", exist_ok=True)

def map_image_to_marker(frame, marker_corners, overlay_img, scale=2.0, offset_ratio=(1.1, 0)):
    if overlay_img.shape[2] == 4:
        # Separate color and alpha channels
        overlay_rgb = overlay_img[:, :, :3]
        overlay_alpha = overlay_img[:, :, 3]
    else:
        overlay_rgb = overlay_img
        overlay_alpha = np.ones(overlay_rgb.shape[:2], dtype=np.uint8) * 255

    src_h, src_w = overlay_rgb.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    dst = marker_corners[0].astype(np.float32)
    top_left, top_right, bottom_right, bottom_left = dst

    right_vec = top_right - top_left
    marker_width = np.linalg.norm(right_vec)
    aspect_ratio = src_h / src_w
    scaled_width = marker_width * scale
    scaled_height = scaled_width * aspect_ratio

    unit_right = right_vec / np.linalg.norm(right_vec)
    down_vec = bottom_left - top_left
    unit_down = down_vec / np.linalg.norm(down_vec)
    unit_perp = np.array([-unit_right[1], unit_right[0]])

    offset = unit_right * (marker_width * offset_ratio[0]) + unit_down * (marker_width * offset_ratio[1])
    new_top_left = top_right + offset
    new_top_right = new_top_left + unit_right * scaled_width
    new_bottom_right = new_top_right + unit_perp * scaled_height
    new_bottom_left = new_top_left + unit_perp * scaled_height

    dst_pts = np.array([new_top_left, new_top_right, new_bottom_right, new_bottom_left], dtype=np.float32)
    src_pts = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_rgb = cv2.warpPerspective(overlay_rgb, H, (frame_w, frame_h))
    warped_alpha = cv2.warpPerspective(overlay_alpha, H, (frame_w, frame_h))

    # Normalize alpha to range 0.0 to 1.0
    alpha_mask = warped_alpha.astype(float) / 255.0
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    # Blend frame and warped image using alpha
    frame = frame.astype(float)
    blended = frame * (1 - alpha_mask) + warped_rgb.astype(float) * alpha_mask
    return blended.astype(np.uint8)

def video_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                overlay_img = marker_images.get(marker_id)
                if overlay_img is not None:
                    frame = map_image_to_marker(frame, corner, overlay_img, scale=2.5)

        with frame_lock:
            latest_frame = frame.copy()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/frame")
async def get_frame():
    with frame_lock:
        if latest_frame is None:
            return Response(content=b"", media_type="image/jpeg")
        _, jpeg = cv2.imencode(".jpg", latest_frame)
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.post("/take_photo")
async def take_photo():
    with frame_lock:
        if latest_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"photos/photo_{timestamp}.png"
            cv2.imwrite(filename, latest_frame)
            return {"status": "saved", "path": filename}
    return {"status": "error"}

if __name__ == "__main__":
    threading.Thread(target=video_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
