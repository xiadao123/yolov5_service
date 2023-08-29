from io import BytesIO
import onnxruntime
import numpy as np
import cv2
from .orientation import non_max_suppression, tag_images


class ONNXModel(object):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path,  providers=['CPUExecutionProvider'])
        # providers = ['CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider']
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed


class YOLO(ONNXModel):
    def __init__(self, onnx_path="sas_slash_20230316.onnx"):
        super(YOLO, self).__init__(onnx_path)
        # 训练所采用的输入图片大小
        self.img_size = 640
        self.img_size_h = self.img_size_w = self.img_size
        self.batch_size = 1
        self.num_classes = 1
        self.classes = ['slash']


    def to_numpy(self, image, shape):
        def letterbox_image(image, size):
            # iw, ih = image.size
            ih, iw, _ = image.shape
            w, h = size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            # PIL
            # image = image.resize((nw, nh), Image.BICUBIC)
            # new_image = Image.new('RGB', size, (128, 128, 128))
            # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
            dw = (w - nw) // 2
            dh = (h - nh) // 2
            new_image[dh:dh + nh, dw:dw + nw, :] = image
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            # cv2.imshow("new_image", new_image)
            # cv2.waitKey(0)
            return new_image

        resized = letterbox_image(image, (self.img_size_w, self.img_size_h))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in

    def decect(self, image):
        # 图片转换为矩阵
        image_numpy = self.to_numpy(image, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        pred = non_max_suppression(outputs[0])
        if pred:
            res = tag_images(image, pred, self.img_size, self.classes)
        else:
            res = []
        return res

