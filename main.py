import cv2
from ultralytics import YOLO
import numpy as np

# import torch

# check if the MPS ins available
# this is the cuda equivalent of
# this should print true if the MPS is available
#print(torch.backends.mps.is_available())

# Load the video
# use any video file you want
cap = cv2.VideoCapture("bigbuck.mp4")

# Load the model
model = YOLO("yolov10n.pt")

# till the video is read correctly
while True:
    # Read the frame
    ret, frame = cap.read()

    # If the frame is not read correctly, the loop will break
    if not ret:
        break

    # Perform inference and get the results
    # The results will be a list of dictionaries
    # Each dictionary will have the keys: boxes, scores, names
    # boxes: list of bounding boxes
    # scores: list of confidence scores
    # names: list of class names
    # set the device to "mps" to use the MPS
    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(),dtype=int)
    classes = np.array(result.boxes.cls.cpu(),dtype=int)

    # Draw bounding boxes on the frame
    for bbox,cls in zip(bboxes,classes):
        (x,y,x2,y2) = bbox
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,225), 2)
        cv2.putText(frame, str(result.names[cls]), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),2)
        

    # Display the frame
    cv2.imshow("Img", frame)

    # Wait for 1ms
    key = cv2.waitKey(1)

    # Press Esc to exit
    if key == 27:
        break

# Release the VideoCapture object and close all the windows
cap.release()
cv2.destroyAllWindows()