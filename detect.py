import cv2
from .operation import YOLO
import os
import time
from typing import List
from numpy import ndarray


def detect(image: ndarray, onnx_path = "/algorithms/OCR/sas_slash_detect/sas_slash_20230316.onnx", show=False) -> List:
    '''
    检测目标，返回目标所在坐标如：
    [{'bbox_xyxy': [57, 390, 207, 882], 'classes': 'slash', "confidence": "0.99"},...]
    :param onnx_path:onnx模型路径
    :param image:检测用的图片
    :param show:是否展示
    :return:
    '''
    yolo = YOLO(onnx_path=onnx_path)
    det_obj = yolo.decect(image)

    # 画框框
    if show:
        for i in range(len(det_obj)):
            top_x, top_y, bottom_x, bottom_y = det_obj[i]['bbox_xyxy']
            cv2.rectangle(image, (int(top_x), int(top_y)), (int(bottom_x), int(bottom_y)), (0, 0, 0), thickness=2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    return det_obj