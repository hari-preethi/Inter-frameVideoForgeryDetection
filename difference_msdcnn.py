import cv2

from preprocess import video_to_grayscale

#Finds difference of two input frames
def difference (frame1, frame2):
    frame_difference = frame1 - frame2
    return frame_difference

#Difference layer of FDCNN
def forward_difference(video):
    fdiff = []
    for i in range (0,(len(video)-1)):
        img1 = video[i]
        img2 = video[i+1]
        frame_difference = difference(img1,img2)
        fdiff.append(frame_difference)
    return fdiff

#Difference layer of PDCNN
def post_difference(video_feature):
    pdiff = []
    for i in range (0,(len(video_feature)-1)):
        img1 = video_feature[i]
        img2 = video_feature[i+1]
        feature_difference = difference(img1,img2)
        pdiff.append(feature_difference)
    return pdiff

#Display differenced frames of FDCNN
def display_forward_difference(video):
    fdiff = forward_difference(video)
    for i in range (0,len(fdiff)):
        cv2.imshow("image",fdiff[i])
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

#Display differenced frames of PDCNN
def display_post_difference(video_feature):
    pdiff = post_difference(video_feature)
    print(pdiff)  



if __name__=='__main__':
    input_video_path = 'D:/Final_Project/Dataset/ForgeryDataset/Deletion/Training/Original/original_train (20).avi'
    video = video_to_grayscale(input_video_path)
    display_forward_difference(video)
    display_post_difference(video)