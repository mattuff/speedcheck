
import torch as T
import torchvision as tv

from flask import Flask, request, render_template

import cv2

import os

from models import Model


model_state = T.load('model')


device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
model_state = T.load('model')
model = Model()
model.load_state_dict(model_state)
model = model.to(device).eval()

transform = tv.transforms.Compose([
    tv.transforms.CenterCrop((720, 720)).to(device),
    tv.transforms.Resize((224, 224)).to(device),
    tv.transforms.ConvertImageDtype(T.float).to(device)
])

raft_weights = tv.models.optical_flow.Raft_Large_Weights.DEFAULT
raft_transforms = raft_weights.transforms().to(device)
raft = tv.models.optical_flow.raft_large(weights=raft_weights, progress=False)
raft = raft.to(device).eval()


def optical_flow(frames1, frames2):
    frames1 = frames1.to(device)
    frames2 = frames2.to(device)
    t_list = list()
    with T.no_grad():
        for img1, img2 in zip(frames1, frames2):
            img1 = T.unsqueeze(img1, 0)
            img2 = T.unsqueeze(img2, 0)
            img1, img2 = raft_transforms(img1, img2)
            t_list.append(raft(img1, img2)[-1])
    return T.squeeze(T.stack(t_list))


grayscale = tv.transforms.Grayscale().to(device)

normalize = tv.transforms.Normalize(
    T.tensor([0.3002, -0.0349, -0.0826]),
    T.tensor([0.2623, 4.1885, 2.1467])
).to(device)


app = Flask(__name__)


def transform_video(video_path, frame_freq):
    vid, _, _ = tv.io.read_video(video_path)
    vid = T.permute(vid, (0, 3, 2, 1))
    vid = transform(vid).to(device)
    frame_idx_list = list(range(16, len(vid), frame_freq))
    tensor_list = list()
    for idx in frame_idx_list:
        frames1 = vid[idx-16: idx: 2, :, :, :]
        frames2 = vid[idx-15: idx+1: 2, :, :, :]
        flow = optical_flow(frames1, frames2)
        gray = grayscale(frames1)
        output = T.cat([gray, flow], dim=1)
        output = normalize(output)
        output = T.permute(output, (1, 0, 2, 3)).unsqueeze(0)
        output = output.to(device)
        tensor_list.append((idx, output))
    return tensor_list


def get_prediction(video_path, frame_freq):
    tensor_list = transform_video(video_path, frame_freq)
    with T.no_grad():
        pred_dict = {idx: model(tensor).item() for idx, tensor in tensor_list}
    return pred_dict


def write_prediction(pred_dict, in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'VP80'),
                          30, (frame_width, frame_height))
    frame_idx = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            pred_idx = frame_idx // 16 * 16
            text = pred_dict.get(pred_idx)
            if text is None:
                text = pred_dict.get(pred_idx + 16)
            if text is None:
                text = pred_dict.get(pred_idx - 16)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.rectangle(frame, (40, 16), (160, 60), (0, 0, 0), -1)
            cv2.putText(frame, f'{text:.1f}', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            out.write(frame)
            frame_idx += 1
        else:
            break

    cap.release()
    out.release()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'in_file' in os.listdir('static'):
            os.remove(os.path.join('static', 'in_file'))
        if 'out_file.webm' in os.listdir('static'):
            os.remove(os.path.join('static', 'out_file.webm'))
        file_upload = request.files['file']
        file_upload.save(os.path.join('static', 'in_file'))
        pred_dict = get_prediction(os.path.join('static', 'in_file'), 16)
        write_prediction(pred_dict, os.path.join('static', 'in_file'), os.path.join('static', 'out_file.webm'))
        return render_template('index.html', filename='out_file.webm')
