import os
import time
import cv2
import json
import torch
from loguru import logger
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from pypylon import pylon

# Instantiate the pylon DeviceInfo object and use it to get the cameras
# tl_factory = pylon.TlFactory.GetInstance()
# devices = tl_factory.EnumerateDevices()
# if len(devices) < 1:
#     raise ValueError("Not enough Pylon cameras found, at least 1 required")

# # Create Pylon camera object
# camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

# # Open the Pylon camera
# camera1.Open()

# # Set up camera resolutions
# camera1.Width.Value = 3000
# camera1.Height.Value = 2000

# # Start grabbing from the Pylon camera
# camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# # Converting Pylon image to OpenCV format
# converter = pylon.ImageFormatConverter()
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Initialize the laptop camera using OpenCV
laptop_camera = cv2.VideoCapture(0)  # 0 is usually the built-in laptop camera


class Predictor(object):
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


def main():
    # Initialize parameters
    experiment_name = None
    model_name = None
    exp_file = "/YOLOX/exps/example/custom/yolox_s.py"
    ckpt_file = "/YOLOX/last_epoch_ckpt.pth"
    device = "gpu"
    conf = 0.5
    nms = 0.3
    tsize = None
    fp16 = False
    legacy = False
    fuse = False
    trt = True  # Using TensorRT

    # Get the experiment configuration
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

    pTime = 0

    while True:
        # Pylon camera
        # grabResult1 = camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Laptop camera
        ret, laptop_frame = laptop_camera.read()

        # if grabResult1.GrabSucceeded() and ret:
        if ret:
            # Process Pylon camera frame
            # image1 = converter.Convert(grabResult1)
            # img1 = image1.GetArray()
            # resized_img1 = cv2.resize(img1, (1900, 1500))

            # Process laptop camera frame
            resized_laptop_frame = cv2.resize(laptop_frame, (1900, 1500))

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - pTime)
            pTime = curr_time

            try:
                # Inference on Pylon camera frame
                # outputs1, img_info1 = predictor.inference(resized_img1)
                # annotated_frame1 = predictor.visual(
                #     outputs1[0], img_info1, predictor.confthre
                # )

                # Inference on laptop camera frame
                outputs2, img_info2 = predictor.inference(resized_laptop_frame)
                annotated_frame2 = predictor.visual(
                    outputs2[0], img_info2, predictor.confthre
                )

                # Display frames
                # cv2.putText(
                #     annotated_frame1,
                #     f"FPS: {int(fps)}",
                #     (0, 50),
                #     cv2.FONT_HERSHEY_PLAIN,
                #     3,
                #     (255, 255, 0),
                #     3,
                # )
                cv2.putText(
                    annotated_frame2,  # type: ignore
                    f"FPS: {int(fps)}",
                    (0, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 255, 0),
                    3,
                )  # type: ignore

                # cv2.imshow("YOLOX Inference Camera 1", annotated_frame1)
                cv2.imshow("YOLOX Inference Laptop Camera", annotated_frame2)  # type: ignore

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                print(e)

    cv2.destroyAllWindows()
    laptop_camera.release()


if __name__ == "__main__":
    main()
