import os
import cv2
import numpy as np

def visualize(data, video_path):
    """Visualize video with AlphaPose bounding box

    Args:
        data (list): AlphaPose result read from a JSON file
        video_path (str): path to the video
    
    # example of generating 'data'
    file_path = "D:\ASH\datasets\Le2i\Coffee_room_all\Skeletons\\video (1).json"
    with open(file_path, 'r') as json_file:
        data_str = json_file.read()
    data = eval(data_str)
    """
    # Initialize video capture from a video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Loop through each frame of the video
    f_idx = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break

        if f_idx >= 45 and f_idx <= 80: 
            if f_idx - 9 >= 0:
                xmin, ymin, width, height = map(int, data[f_idx]['box'])
                # Draw the rectangle on the image
                cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
            cv2.imshow('Video with Bounding Boxes', frame)
        else: 
            height, width, _ = frame.shape
            cv2.imshow('Video with Bounding Boxes', np.zeros((height, width, 3), dtype=np.uint8))
            
        f_idx += 1

        # Wait for a key press and check if it's the 'q' key to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def split_fall_seq(filename, seq, start, end, nframes):
    # seq.shape == (num_frames, 17, 2)
    
    pivot = start
    
    left = []
    for i in range(0, pivot, nframes):
        end_frame = min(pivot, i + nframes, len(seq))
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
        subseq = seq[start_frame:end_frame]
        if len(subseq) != nframes:
            print(start_frame, end_frame, len(seq), pivot, start, filename)
            print(len(subseq))
        left.append(subseq)
    
    right = []
    for i in range(pivot + nframes, len(seq), nframes):
        end_frame = min(len(seq), i + nframes)
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
        subseq = seq[start_frame:end_frame]
        right.append(subseq)
    
    end_frame = min(len(seq), start + nframes)
    start_frame = start
    if end_frame - start_frame != nframes:
        start_frame -= nframes - (end_frame - start_frame)
        if start_frame < 0:
            offset = 0 - start_frame
            start_frame += offset
            end_frame += offset
    fall_seq = seq[start_frame:end_frame] # the subseq of the fall event
    
    non_falls = left + right # a list of subseqs of all the non-fall events
    
    return fall_seq, non_falls

def split_seq(seq, nframes):
    res = []
    for i in range(0, len(seq), nframes):
        end_frame = min(len(seq), i + nframes)
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
        subseq = seq[start_frame:end_frame]
        res.append(subseq)
    
    return res

def add_flows(flows, subseqs, start_offset, nframes):
    new_subseqs = []
    for subseq in subseqs:
        start = subseq['start_frame'] + start_offset
        new_subseq = subseq.copy()
        new_subseq['flows'] = flows[start:start + nframes]
        if start + nframes - 1 == len(flows):
            new_subseq['flows'] = np.concatenate((new_subseq['flows'], flows[-1].reshape(1, -1)), axis=0)
        if len(new_subseq['flows']) == nframes:
            new_subseqs.append(new_subseq)
    return new_subseqs
    
def split_skeletons(skeletons, nframes=45, flow_path=None):
    # skeletons is an array of preprocessed skeleton dicts with keys filename, keypoints, scores, boxes, fall_interval, offset
    res_falls = []
    res_non_falls = []
    for seq in skeletons:
        filename = seq['filename']
        
        # start and end frame of the fall event in the skeleton sequence
        offset = seq['offset']
        start_frame, end_frame = seq['fall_interval']
        
        if start_frame == 0 and end_frame == 0:
            non_fall = split_seq(seq['keypoints'], nframes)
        else:
            fall, non_fall = split_fall_seq(filename, seq['keypoints'], start_frame, end_frame, nframes)
            res_falls.append(fall)
        
        # if flow_path:
        #     file_path = os.path.join(flow_path, filename + '.npy')
        #     flows = np.load(file_path)
        #     fall = add_flows(flows, [fall], offset, nframes)[0]
        #     non_fall = add_flows(flows, non_fall, offset, nframes)
            
        res_non_falls += non_fall
    
    return res_falls, res_non_falls