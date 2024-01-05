# Powerful-IoU
Powerful-IoU：More straightforward and faster bounding box regression loss with a nonmonotonic focusing mechanism

This is the source code for Powerful-IoU.
Link to article:
https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640



using Powerful-IoU in Yolov8:
1.Download the source code for Yolov8. Link: https://github.com/ultralytics/ultralytics
2.Copy and paste the source code from our file YOLOv8/piou.py into ultralytics-main/ultralytics/yolo/utils/metrics.py
3.Add code to ultralytics-main/ultralytics/yolo/utils/loss.py
from .metrics import piou
4.Modify the use of the loss function in class BboxLoss in ultralytics-main/ultralytics/yolo/utils/loss.py
iou =1-piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True)
