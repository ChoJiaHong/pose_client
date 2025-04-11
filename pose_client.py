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
        channel = grpc.insecure_channel(server)
        self.stub = pose_pb2_grpc.MirrorStub(channel)

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


def process_images(image_path):
    client = PoseDetectionClient('10.52.52.25:' + '30500')

    response = client.send_image(image_path)
    if response:
        print(f"Received Skeletons for {image_path}: {response.skeletons}")
        print(type(response.skeletons))


if __name__ == "__main__":
    #while True:
    process_images('1280.jpg')
