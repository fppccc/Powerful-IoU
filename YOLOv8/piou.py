

def piou(box1, box2, xywh=True, PIoU=False,PIoU2=False,Lambda=1.3,eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union   
    
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height


    # PIoU
    dw1=torch.abs(b1_x2.minimum(b1_x1)-b2_x2.minimum(b2_x1))
    dw2=torch.abs(b1_x2.maximum(b1_x1)-b2_x2.maximum(b2_x1))
    dh1=torch.abs(b1_y2.minimum(b1_y1)-b2_y2.minimum(b2_y1))
    dh2=torch.abs(b1_y2.maximum(b1_y1)-b2_y2.maximum(b2_y1))
    P=((dw1+dw2)/torch.abs(w2)+(dh1+dh2)/torch.abs(h2))/4
    L_v1=1-iou - torch.exp(-P**2)+1
    
    
    if PIoU:        
        return L_v1 

    if PIoU2:    
        q=torch.exp(-P)
        x=q*Lambda
        return  3*x*torch.exp(-x**2)*L_v1


