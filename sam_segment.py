from ultralytics import SAM
import numpy as np
import cv2

model = SAM("sam2_b.pt")

#display the model info
model.info()

#Segment everything
results = model("test.jpg")

img = cv2.imread("test.jpg")

height, width, channels = img.shape

#Get the first result
result = results[0]

#Get the bounding boxes
bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

segments = []

for seg in result.masks.xyn:
    # contours
    seg[:, 0] *= width
    seg[:, 1] *= height
    segment = np.array(seg, dtype=np.int32)
    segments.append(segment)

#draw bounding boxes on the image
for bbox,seg in zip(bboxes,segments):
    (x, y, x2, y2) = bbox
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 225), 2)
    cv2.polylines(img,[seg], True, (0, 255, 0), 2)

#display the image
cv2.imshow("Image", img)
cv2.waitKey(0)

#display the results
print(results)