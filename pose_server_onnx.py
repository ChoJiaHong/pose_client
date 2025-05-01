import grpc
from concurrent import futures
import time
import cv2
import numpy as np
import json
import queue
import threading
import onnxruntime as ort

from config import settings
import pose_pb2
import pose_pb2_grpc
from grpc_health.v1 import health_pb2_grpc, health_pb2


# --- 工具函式區 ---
def preprocess_image(image_data):
    img_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


def preprocess_batch(frames):
    processed = []
    for frame in frames:
        img = cv2.resize(frame, (settings.img_size, settings.img_size))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        processed.append(img)
    return np.stack(processed, axis=0)


def postprocess_results(outputs):
    preds = outputs[0]  # 取第一個輸出 (通常是推理結果)
    batch_skeletons = []
    for pred in preds:
        skeletons = []
        # 這裡需要根據你的 ONNX 輸出格式解析
        # 下面是假設 pred 是 (num_keypoints, 3): (x, y, confidence)
        for idx, kpt in enumerate(pred):
            x, y, conf = kpt
            if conf >= 0.5:
                skeletons.append((int(x), int(y), round(float(conf), 3)))
        batch_skeletons.append(json.dumps([{"id": idx, "keypoints": skeletons}]))
    return batch_skeletons


class ONNXPoseModel:
    def __init__(self, model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch_frames):
        inputs = preprocess_batch(batch_frames)
        outputs = self.session.run(None, {self.input_name: inputs})
        return postprocess_results(outputs)


# --- gRPC服務區 ---
class HealthServicer(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)


class RequestWrapper:
    def __init__(self, image_data):
        self.image_data = image_data
        self.result_queue = queue.Queue()


class PoseDetectionWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.pose_model = ONNXPoseModel(settings.onnx_weights)
        self.queue = queue.Queue()
        self.batch_size = settings.batch_size
        self.queue_timeout = settings.queue_timeout
        threading.Thread(target=self._batch_worker, daemon=True).start()

    def handle_request(self, request):
        wrapper = RequestWrapper(request.image_data)
        self.queue.put(wrapper)
        return wrapper.result_queue.get()

    def _batch_worker(self):
        while True:
            batch_frames = []
            wrappers = []
            start_time = time.time()
            while len(batch_frames) < self.batch_size and (time.time() - start_time) < self.queue_timeout:
                try:
                    wrapper = self.queue.get(timeout=0.01)
                    frame = preprocess_image(wrapper.image_data)
                    batch_frames.append(frame)
                    wrappers.append(wrapper)
                except queue.Empty:
                    pass

            if not batch_frames:
                continue

            try:
                outputs = self.pose_model.predict(batch_frames)
                for output, wrapper in zip(outputs, wrappers):
                    wrapper.result_queue.put(output)
            except Exception as e:
                print(f"[Worker-{self.worker_id}] Error:", e)
                for wrapper in wrappers:
                    wrapper.result_queue.put("")


class PoseDetectionService(pose_pb2_grpc.MirrorServicer):
    def __init__(self):
        self.num_workers = settings.num_workers
        self.workers = [PoseDetectionWorker(worker_id=i) for i in range(self.num_workers)]
        self.next_worker = 0
        self.lock = threading.Lock()

    def SkeletonFrame(self, request, context):
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % self.num_workers
        result = worker.handle_request(request)
        return pose_pb2.FrameResponse(skeletons=result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    pose_pb2_grpc.add_MirrorServicer_to_server(PoseDetectionService(), server)
    server.add_insecure_port('[::]:' + settings.gRPC_port)
    server.start()
    print(f"[pose] gRPC server (multi-model instance, ONNX backend) running on port {settings.gRPC_port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()