import os
import torch
import argparse
import numpy as np

from model.stgcn import stgcn, flow_stgcn
from model.st_graph import get_distance_adjacency
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

HOME_PATH = "D:\ASH\datasets\Le2i\Home_all\Skeletons_full"
OFFICE_PATH = "D:\ASH\datasets\Le2i\Office\Skeletons_split"
LEC_ROOM_PATH = "D:\ASH\datasets\Le2i\Lecture room\Skeletons_split"

HOME_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
OFFICE_LABELS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
LEC_ROOM_LABELS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

default_layer_config = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                        (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)] # (in_channles, out_channels, temporal_stride)

edges = [
    [0, 1],  # Nose - Left Eye
    [0, 2],  # Nose - Right Eye
    [1, 3],  # Left Eye - Left Ear
    [2, 4],  # Right Eye - Right Ear
    [1, 5],  # Left Eye - Left Shoulder
    [2, 6],  # Right Eye - Right Shoulder
    [5, 7],  # Left Shoulder - Left Elbow
    [6, 8],  # Right Shoulder - Right Elbow
    [7, 9],  # Left Elbow - Left Wrist
    [8, 10],  # Right Elbow - Right Wrist
    [5, 11],  # Left Shoulder - Left Hip
    [6, 12],  # Right Shoulder - Right Hip
    [11, 13],  # Left Hip - Left Knee
    [12, 14],  # Right Hip - Right Knee
    [13, 15],  # Left Knee - Left Ankle
    [14, 16]   # Right Knee - Right Ankle
]

max_x = 320
max_y = 240

def load_data(path):
    data = []
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            seq = np.load(file_path, allow_pickle=True).item()
            data.append(seq['keypoints'])
    return data

def create_batch(frames, stride=1, segment_length=45):
    batch = []
    start = 0
    while start + segment_length <= len(frames):
        end = start + segment_length
        batch.append(frames[start:end, :, :])
        start += stride
    
    if start < len(frames):
        end = len(frames)
        start = end - segment_length
        batch.append(frames[start:end, :, :])

    batch = np.array(batch)
    batch = batch[:, :, :, :2].astype(np.float32)
    batch[:, :, :, 0] /= max_x
    batch[:, :, :, 1] /= max_y
    batch = np.transpose(batch, (0, 3, 1, 2))
    
    return batch

def evaluate(model, device, testset='lecture room'):
    model.eval()
    
    if args.testset == 'home':
        data = load_data(HOME_PATH)
        labels = HOME_LABELS
    elif args.testset == 'office':
        data = load_data(OFFICE_PATH)
        labels = OFFICE_LABELS
    else:
        data = load_data(LEC_ROOM_PATH)
        labels = LEC_ROOM_LABELS
    
    acc = 0
    for i in tqdm(range(len(data))):
        frames = data[i] # (num_frames, 17, 2)
        
        batch = create_batch(frames)
        batch = torch.tensor(batch).to(device)
        
        # batch = frames[:, :, :2].astype(np.float32)
        # batch = torch.tensor(batch).unsqueeze(0)
        # batch[:, :, :, 0] /= max_x
        # batch[:, :, :, 1] /= max_y
        # batch = batch.permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            y_pred = model(batch)
            
            if labels[i] == 1:
                y_batch = torch.ones(len(batch)).to(device)
            else:
                y_batch = torch.zeros(len(batch)).to(device)
                
            if torch.any(torch.argmax(y_pred, axis=-1) == y_batch).detach().cpu().item():
                pred = 1
            else:
                pred = 0
            acc += pred
    return acc / len(data)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True,
                    help='path to the model weights')
    parser.add_argument('--testset', type=str, default='lecture room',
                    help='which test set to use, can be office or lecture room')
    parser.add_argument('--dst', type=str, default='./',
                    help='path to the destination folder')
    
    args = parser.parse_args()
        
    num_node = 17
    A = get_distance_adjacency(np.array(edges), num_node)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = stgcn(num_class=2, window_size=45, num_point=17, graph=A, layer_config=default_layer_config)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    print(f"Evaluating on set: {args.testset}")
    accuracy = evaluate(model, device, args.testset)
    print(accuracy)