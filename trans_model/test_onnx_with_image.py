import onnxruntime as ort
import numpy as np
import cv2
import sys

# ==== 修改這裡 ====
onnx_path = "yolov8n-pose.onnx"           # 你的 ONNX 模型路徑
image_path = "/home/user/jiahong/code/tensor_model/pose_client/1280.jpg"     # 你的圖片路徑
input_size = (640, 640)           # 根據模型實際輸入大小
# ===================

# 載入圖片
frame = cv2.imread(image_path)
if frame is None:
    print(f"[ERROR] 無法讀取圖片：{image_path}")
    sys.exit(1)

# 前處理（跟 YOLOv8 類似的格式）
img = cv2.resize(frame, input_size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1).astype(np.float32)  # HWC → CHW
img /= 255.0
img = np.expand_dims(img, axis=0)  # Add batch dim

# 載入 ONNX 模型
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 推論
outputs = session.run(None, {input_name: img})
print(f"[INFO] 輸出數量: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output[{i}]: shape={output.shape}, dtype={output.dtype}")
    print("前幾個值:", output.flatten()[:10])
