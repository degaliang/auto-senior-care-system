import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from sklearn.decomposition import PCA


import os
for dirname, _, filenames in os.walk('/kaggle/input'): #TYPE PATH TO DATASET
    for filename in filenames:
        os.path.join(dirname, filename)


def flows_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        cap.release()
        exit()
    # Initial Frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        cap.release()
        exit()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow_matrices = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break 
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_matrices.append(flow)
        prev_gray = next_gray
    cap.release()
    return flow_matrices


lecture_1_flows = flows_from_video('/kaggle/input/falldataset-imvia/Lecture_room/Lecture room/video (1).avi') #TYPE PATH TO VIDEO OF CHOICE
print(np.shape(lecture_1_flows))

lecture_1_flows_array = np.array(lecture_1_flows)

num_frames, height, width, num_components = lecture_1_flows_array.shape

# Flatten the spatial dimensions and keep the flow components together
optical_flow_flat = lecture_1_flows_array.reshape(num_frames, height * width * num_components)

# Initialize PCA
pca = PCA(n_components=0.95) 

reduced_flows = pca.fit_transform(optical_flow_flat)
print(reduced_flows.shape)