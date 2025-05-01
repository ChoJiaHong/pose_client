import onnxruntime
import numpy as np

# 建立 ONNX 推論 session
session = onnxruntime.InferenceSession("./yolov8n-pose.onnx", providers=['CUDAExecutionProvider'])

# 測試資料
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = {"input": input_data}

# 推論
outputs = session.run(None, inputs)
print("推論完成，結果形狀：", outputs[0].shape)
