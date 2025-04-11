import grpc
from concurrent import futures
import time
import pose_pb2
import pose_pb2_grpc
import cv2
import numpy as np
from ultralytics import YOLO
from config import settings
import json
from grpc_health.v1 import health_pb2_grpc, health_pb2

class HealthServicer(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

class PoseDetectionService(pose_pb2_grpc.MirrorServicer):
    def __init__(self):
        self.yolo_model = YOLO(settings.weights)

    def SkeletonFrame(self, request, context):
        try:
            # 解碼圖片
            img_data = np.frombuffer(request.image_data, np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            # 使用YOLO模型進行推論
            yolo_results = self.yolo_model(frame, device=settings.device, conf=settings.conf_thres, iou=settings.iou_thres)

            # 轉換 YOLO 結果為 landmarks 格式
            skeletons = []
            for i, yolo_result in enumerate(yolo_results[0]):
                landmarks = self.yolo_result2landmarks(yolo_result.keypoints)
                skeletons.append({"id": i, "keypoints": landmarks})

            return pose_pb2.FrameResponse(skeletons=json.dumps(skeletons))

        except Exception as e:
            print(e)
            return pose_pb2.FrameResponse(skeletons="")

    def yolo_result2landmarks(self, kpts):
        kpts_xy = kpts[0].xy.cpu().numpy()
        num_kpts = kpts_xy.shape[1]
        kpts_conf = kpts[0].conf.cpu().numpy()

        landmarks = []
        for kid in range(num_kpts):
            x_coord, y_coord = int(kpts_xy[0][kid][0]), int(kpts_xy[0][kid][1])
            conf = kpts_conf[0][kid]

            if conf < 0.5:
                continue

            landmarks.append((int(x_coord), int(y_coord), round(float(conf), 3)))
        
        return landmarks


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    pose_pb2_grpc.add_MirrorServicer_to_server(PoseDetectionService(), server)
    server.add_insecure_port('[::]:' + settings.gRPC_port)
    server.start()
    print(f"Server running on port {settings.gRPC_port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
