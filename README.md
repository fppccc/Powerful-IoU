# ***Powerful-IoU：More straightforward and faster bounding box regression loss with a nonmonotonic focusing mechanism***

This is the source code for __Powerful-IoU__.
[[**Article Link**](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)]
<br />
<br />
### Using Powerful-IoU in YOLOv8:

1. Download the source code for YOLOv8. [[**YOLOv8**](https://github.com/ultralytics/ultralytics)] <br />

2. Copy and paste the source code from our file YOLOv8/piou.py into ultralytics-main/ultralytics/yolo/utils/metrics.py<br />

3. Add code to ultralytics-main/ultralytics/yolo/utils/loss.py<br />
```
    from .metrics import piou
```
<br />

4. Modify the use of the loss function in ```class BboxLoss``` in ultralytics-main/ultralytics/yolo/utils/loss.py<br />
```
    iou = 1-piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True)
```
    If the performance of PIOUv2 is unsatisfactory, you may consider trying the following code.
```
    iou = 1-piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True).clamp_(0)
```

