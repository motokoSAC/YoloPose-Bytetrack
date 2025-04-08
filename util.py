
def match_bounding_boxes(model_box, bytetrack_boxes, iou_threshold=0.4):
    match_id = -1
    max_iou = 0
    for id, bytetrack_box in bytetrack_boxes.items():  # dict
        iou = calculate_iou(bytetrack_box, model_box)
        if iou > max_iou:
            max_iou = iou
            match_id = id
    if max_iou > iou_threshold and match_id != -1:
        # bytetrack_boxes.pop(match_id)
        return match_id
    else:
        return -1

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = map(int, box1[:4])
    x1_, y1_, x2_, y2_ = map(int, box2[:4])

    # 计算交集的坐标
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    # 计算交集的面积
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # 计算两个边界框的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # 计算并集的面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


