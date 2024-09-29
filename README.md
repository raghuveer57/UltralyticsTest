### Summary

This contains the code to run inference on a video ( preferably an .mp4 file ) using any of the yolo models

### Usage

Replace the video file path in the line 14
```
cap = cv2.VideoCapture("bigbuck.mp4")
```

Replace the model that you want to use in the line 17.
( Note that the model is downloaded automatically, you need not download the model and place it, just renaming to the available model will download it)

```
model = YOLO("yolov10n.pt")
```
Check the official ultralytics page to try out the available models

Set the device param to make the most out of the available hardware
```
results = model(frame, device="mps")
```
Device can be set to GPU or CPU. It is set to mps while using mac mini.

### Tips

Take some time to understand the results that are obtained from the model.
It has more data that what is documented and can be used as per many use cases.
