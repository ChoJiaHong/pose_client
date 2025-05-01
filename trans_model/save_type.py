import torch

model = torch.load("../yolov8n-pose.pt")
print(type(model))
