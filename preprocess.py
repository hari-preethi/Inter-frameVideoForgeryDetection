import cv2
import numpy as np

def video_to_grayscale(input_path):

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = []
    count = 0
    
    while count<frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        video.append(img)
        
#Display resized grayscale videos
    for i in range (0,len(video)):
        cv2.imshow("image",video[i])
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    print(len(video))
    return video


def video_to_grayscale_extract_frames(input_path, num_frames):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print("Original frame count ", frame_count)
    video = []
    count = 0
    
    total_frames = min(frame_count, num_frames)
    frame_interval = max(frame_count // total_frames, 1)
    out_frame_count = 0

    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        video.append(img)
        
        # Increment output frame count and break if reached desired number of frames
        out_frame_count += 1
        if out_frame_count >= total_frames:
            break
        
    cap.release()
    
    return np.stack(video)




# Example pre-processed video
if __name__=='__main__':
    input_video_path = "D:/Final_Project/Dataset/ForgeryDataset/Deletion/Training/Original/original_train (10).avi"
    input_video_path1 = "uploads/vid1.mp4"
    video = video_to_grayscale(input_video_path1)
    #print(video.shape)    
    print(np.array([video]).shape)  

    """extr = video_to_grayscale_extract_frames(input_video_path1,60)
    print(extr.shape)
    print(len(extr))"""