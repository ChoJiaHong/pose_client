FROM ultralytics/ultralytics:latest

# 設置工作目錄
WORKDIR /app

# 複製當前目錄的所有檔案到容器中的工作目錄
COPY config.py pose_pb2_grpc.py pose_pb2.py pose_server.py yolov8n-pose.pt /app/

# 安裝必要的 Python 套件
RUN pip install grpcio grpcio-tools pydantic fastapi uvicorn grpcio-health-checking==1.65.0

# 設置環境變數以防止 Python 生成 pyc 檔案
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 預設運行的指令，當執行容器時將會運行 PoseDetection.py
CMD ["python", "pose_server.py"]
