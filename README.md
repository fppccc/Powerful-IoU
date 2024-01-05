# Powerful-IoU
## ***Powerful-IoU：More straightforward and faster bounding box regression loss with a nonmonotonic focusing mechanism***
<br />
This is the source code for **Powerful-IoU **.<br />
**Link to article: **  <br />
https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640
<br />
<br />

### using Powerful-IoU in Yolov8:

1.Download the source code for Yolov8. Link: https://github.com/ultralytics/ultralytics<br />

2.Copy and paste the source code from our file YOLOv8/piou.py into ultralytics-main/ultralytics/yolo/utils/metrics.py<br />

3.Add code to ultralytics-main/ultralytics/yolo/utils/loss.py<br />
```
    from .metrics import piou
```
<br />

4.Modify the use of the loss function in class BboxLoss in ultralytics-main/ultralytics/yolo/utils/loss.py<br />
```
    iou = 1-piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True)
```
