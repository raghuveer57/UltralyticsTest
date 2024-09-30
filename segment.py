import cv2
from yolo_segmentation import YOLOSegmentation

# Load the test image
img = cv2.imread("test.jpg")

# use this if you want to resize the image
#img = cv2.resize(img, None, fx=0.5, fy=0.5)

# Segmentation detector class
ys = YOLOSegmentation("yolov8m-seg.pt")

(bboxes,classes,segments,scores)= ys.detect(img)
# print (bboxes)

# Draw bounding boxes on the image
for bbox,seg in zip(bboxes,segments):
        (x,y,x2,y2) = bbox
        cv2.rectangle(img,(x,y),(x2,y2),(0,0,225), 2)
        cv2.polylines(img,[seg], True, (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)