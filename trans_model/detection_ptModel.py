from ultralytics import YOLO
import cv2

# === Step 1: 載入 Pose 模型（.pt 檔案）===
model_path = "/home/user/jiahong/code/tensor_model/pose_client/yolov8n-pose.pt"  # 請換成你的模型路徑
model = YOLO(model_path)

# === Step 2: 讀取圖片 ===
image_path = "/home/user/jiahong/code/tensor_model/pose_client/1280.jpg"  # 請換成你要測試的圖片
img = cv2.imread(image_path)



input_size = model.model.args['imgsz']

# 轉換為 tuple 格式 (H, W)
if isinstance(input_size, int):
    input_h = input_w = input_size
else:
    input_h, input_w = input_size

dummy_input_shape = (1, 3, input_h, input_w)
print(f"模型建議的輸入 shape: {dummy_input_shape}")
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/local_repo/nv-tensorrt-local-repo-ubuntu2404-10.9.0-cuda-12.8_1.0-1_amd64.deb