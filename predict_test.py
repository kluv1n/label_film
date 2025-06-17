import torch
import torchvision
from torchvision import transforms, models
from torch import nn
from PIL import Image
import os
import json

from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_folder = '/content/drive/MyDrive/vk_test/frames/test_frames_all'
min_consecutive = 3
fps = 25

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('/content/drive/MyDrive/vk_test/trained_model.pth', map_location=device))
model = model.to(device)
model.eval()

def predict_frame(model, path):
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def detect_intro_segments(preds, min_consecutive=5):
    events = []
    count = 0
    start_idx = None
    for i, val in enumerate(preds):
        if val == 1:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_consecutive:
                events.append((start_idx, i - 1))
            count = 0
            start_idx = None
    if count >= min_consecutive:
        events.append((start_idx, len(preds) - 1))
    return events

def frame_to_time(index, fps):
    seconds = index / fps
    return str(timedelta(seconds=int(seconds)))

results = {}

for folder_name in sorted(os.listdir(base_folder)):
    folder_path = os.path.join(base_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue

    frame_paths = sorted([
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    predictions = [predict_frame(model, path) for path in frame_paths]
    events = detect_intro_segments(predictions, min_consecutive)

    segments = [
        {
            "start_time": frame_to_time(start, fps),
            "end_time": frame_to_time(end, fps)
        } for start, end in events
    ]

    results[folder_name] = segments

with open('/content/drive/MyDrive/vk_test/intro_segments.json', 'w') as f:
    json.dump(results, f, indent=4)
