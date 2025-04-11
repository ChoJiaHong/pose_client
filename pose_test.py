import grpc
import cv2
import pose_pb2
import pose_pb2_grpc
import os
import numpy as np
from config import settings
import time
import asyncio

class PoseDetectionClient:
    def __init__(self, server):
        self.channel = grpc.insecure_channel(server)
        self.stub = pose_pb2_grpc.MirrorStub(self.channel)

    def send_image(self, image_path):
        try:
            # 讀取圖片
            image = cv2.imread(image_path)
            _, img_data = cv2.imencode('.jpg', image)

            # 傳送圖片給Server
            request = pose_pb2.FrameRequest(image_data=img_data.tobytes())
            response = self.stub.SkeletonFrame(request)

            return response

        except Exception as e:
            print(f"Error sending image: {e}")
            return None


async def async_process_images(client, image_path, start_time, response_times):
    """非同步處理單張圖片並計算回應時間"""
    try:
        response = await asyncio.to_thread(client.send_image, image_path)
        if response:
            response_time = time.time() - start_time
            response_times.append(response_time)
    except Exception as e:
        print(f"Error in async process: {e}")


async def run():
    client = PoseDetectionClient('localhost:30510')
    image_path = '1280.jpg'
    
    num_requests = 10000 # 目標發送請求數
    response_times = []  # 儲存每個請求的回應時間
    tasks = []  # 儲存所有請求的非同步任務

    start_time = time.time()  # 記錄開始時間

    # 發送 1000 個非同步請求
    for _ in range(num_requests):
        task = async_process_images(client, image_path, start_time, response_times)
        tasks.append(asyncio.create_task(task))

    # 等待所有請求完成
    await asyncio.gather(*tasks)

    # 記錄第一個與最後一個請求完成的時間
    first_response_time = min(response_times) if response_times else 0
    last_response_time = max(response_times) if response_times else 0

    # 計算 1000 個請求的平均處理時間
    total_duration = last_response_time - first_response_time if response_times else 0
    avg_time_per_request = total_duration / num_requests if num_requests > 0 else 0

    print("\n--- Performance Statistics ---")
    print(f"First Response Time: {first_response_time:.4f} sec")
    print(f"Last Response Time: {last_response_time:.4f} sec")
    print(f"Total Duration: {total_duration:.4f} sec")
    print(f"Average Time per Request: {avg_time_per_request:.6f} sec")


if __name__ == "__main__":
    asyncio.run(run())
