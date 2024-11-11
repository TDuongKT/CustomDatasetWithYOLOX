import math
import os
import cv2
import time
import torch
import imutils
from datetime import datetime, date
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis
from loguru import logger

cwd = os.getcwd()


class Predictor:
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_line_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


dist_front_threshold = 160
dist_back_threshold = 100
dist_back2_threshold = 180


def update_front_threshold(val):
    global dist_front_threshold
    dist_front_threshold = val


def update_back_threshold(val):
    global dist_back_threshold
    dist_back_threshold = val


def update_back2_threshold(val):
    global dist_back2_threshold
    dist_back2_threshold = val


cv2.namedWindow("Parameters", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Parameters", 300, 150)
cv2.createTrackbar(
    "Front Threshold", "Parameters", dist_front_threshold, 500, update_front_threshold
)
cv2.createTrackbar(
    "Back Threshold", "Parameters", dist_back_threshold, 500, update_back_threshold
)
cv2.createTrackbar(
    "Back2 Threshold", "Parameters", dist_back2_threshold, 500, update_back2_threshold
)


def main():
    image_path = "D:/Honda_PlusVN/Python/Python_Project/YOLOX/z5710890611475_2ef4d744fd80dbfe5c8743c0f3feaf75.jpg"
    experiment_name = None
    model_name = None
    exp_file = (
        "D:/Honda_PlusVN/Python/Python_Project/YOLOX/exps/example/custom/yolox_s.py"
    )
    ckpt_file = "D:/Honda_PlusVN/Python/Python_Project/YOLOX/best_filmerorr.pth"
    device = "gpu"
    conf = 0.3
    nms = 0.3
    tsize = None
    fp16 = False
    legacy = False
    fuse = False
    trt = True

    exp = get_exp(exp_file, model_name)
    if experiment_name is None:
        experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    logger.info(f"Running inference on device: {device}")
    logger.info(f"Using TensorRT: {trt}")

    exp.test_conf = conf
    exp.nmsthre = nms
    if tsize is not None:
        exp.test_size = (tsize, tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()
    model.eval()

    if not trt:
        if ckpt_file is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if trt:
        assert not fuse, "TensorRT model does not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model not found! Run conversion script first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder, device, fp16, legacy
    )

    outputs, img_info = predictor.inference(image_path)
    result_image = predictor.visual(outputs, img_info, conf)

    bottleneck = []
    front = []
    back = []

    count = 0
    message = "Waiting To Detect....."

    frame = imutils.resize(result_image, width=1200)
    frame = frame[100:1200, 250:800]

    check_bottleneck = True
    check_front = True
    check_back = True

    check_has_bottleneck = 0

    for obj in outputs[0]:
        name = COCO_CLASSES[int(obj[6])]
        box = obj[0:4].tolist()
        if name == "bottleneck":
            bottleneck.extend(box)
            check_has_bottleneck = 1
            count += 1
        elif name == "front" and check_front:
            front.extend(box)
            count += 1
        elif name == "back" and check_back:
            back.extend(box)
            count += 1

    dist_neck_to_back = 0
    dist_neck_to_front = 0

    center_y2 = 0

    if check_has_bottleneck > 0 and count > 1:
        if len(bottleneck) > 0 and len(front) > 0:
            centroid_bottleneck = (
                int(bottleneck[0] + (bottleneck[2] / 2)),
                int(bottleneck[1] + (bottleneck[3] / 2)),
            )
            centroid_front = (
                int(front[0] + (front[2] / 2)),
                int(front[1] + (front[3] / 2)),
            )

            center_x1, center_y1 = get_line_center(
                int(front[0]), int(front[1]), int(front[2]), int(front[3])
            )
            center_x2, center_y2 = get_line_center(
                int(bottleneck[0]),
                int(bottleneck[1]),
                int(bottleneck[2]),
                int(bottleneck[3]),
            )
            dist_neck_to_front = calculate_distance(
                center_x2, center_y1, center_x2, center_y2
            )

            cv2.putText(
                frame,
                "Front : " + str(dist_neck_to_front),
                (0, 250),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 255),
                3,
            )
            cv2.line(
                frame, (center_x2, center_y1), (center_x2, center_y2), (0, 0, 255), 2
            )
        elif len(bottleneck) > 0 and len(back) > 0:
            centroid_bottleneck = (
                int(bottleneck[0] + (bottleneck[2] / 2)),
                int(bottleneck[1] + (bottleneck[3] / 2)),
            )
            centroid_back = (int(back[0] + (back[2] / 2)), int(back[1] + (back[3] / 2)))

            center_x1, center_y1 = get_line_center(
                int(back[0]), int(back[1]), int(back[2]), int(back[3])
            )
            center_x2, center_y2 = get_line_center(
                int(bottleneck[0]),
                int(bottleneck[1]),
                int(bottleneck[2]),
                int(bottleneck[3]),
            )
            dist_neck_to_back = calculate_distance(
                center_x2, center_y1, center_x2, center_y2
            )

            cv2.putText(
                frame,
                "Back : " + str(dist_neck_to_back),
                (0, 250),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 255),
                3,
            )
            cv2.line(
                frame, (center_x2, center_y1), (center_x2, center_y2), (0, 0, 255), 2
            )

        else:
            message = "Can not detect"

    if dist_neck_to_front > dist_front_threshold:
        message = "Wrong"
    elif 0 < dist_neck_to_front < dist_front_threshold:
        message = "OK"
    elif dist_neck_to_back > dist_back_threshold:
        message = "OK"
    elif 0 < dist_neck_to_back < dist_back_threshold:
        message = "Wrong"

    else:
        message = "Waiting To Detect....."

    fps = 1 / (time.time() - pTime)
    pTime = time.time()

    cv2.putText(
        frame, f"FPS: {int(fps)}", (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3
    )
    cv2.putText(frame, message, (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Result", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
