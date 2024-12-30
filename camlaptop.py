# import os
# import cv2
# import time
# import torch
# from datetime import datetime
# from threading import Thread
# from queue import Queue
# from loguru import logger
# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis
# import shutil


# import signal
# import atexit


# # Hàm gửi tín hiệu tắt còi
# def turn_off_buzzer():
#     try:
#         arduino.write(b"0")  # Gửi tín hiệu '0' để tắt còi
#         print("Buzzer turned off.")
#     except Exception as e:
#         print(f"Failed to turn off buzzer: {e}")


# # Gắn hàm tắt còi vào sự kiện kết thúc chương trình
# atexit.register(turn_off_buzzer)


# # Bắt tín hiệu dừng chương trình (Ctrl+C hoặc đóng chương trình)
# def signal_handler(sig, frame):
#     print("Program is stopping...")
#     turn_off_buzzer()
#     arduino.close()  # Đóng kết nối với Arduino
#     exit(0)


# signal.signal(signal.SIGINT, signal_handler)
# # Khởi tạo camera laptop
# laptop_camera = cv2.VideoCapture(0)

# # Thư mục gốc để lưu ảnh
# BASE_DIR = "Thelf_Detection"

# # Hàng đợi để lưu ảnh
# image_queue = Queue()

# import os
# import shutil
# import cv2

# BASE_DIR = "Thelf_Detection"

# import serial
# import serial.tools.list_ports

# # Tìm cổng COM đúng
# ports = serial.tools.list_ports.comports()
# for port in ports:
#     print(f"Found port: {port.device}")

# try:
#     # Mở cổng COM (thay COM8 bằng cổng đúng)
#     arduino = serial.Serial(port="COM8", baudrate=9600, timeout=1)
#     print("Arduino connected successfully!")
# except serial.SerialException as e:
#     print(f"Failed to connect: {e}")


# def send_command_to_arduino(command):

#     arduino.write(command.encode())  # Gửi tín hiệu đến Arduino


# def save_image(frame):
#     temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
#     final_path = os.path.join(BASE_DIR, "image.jpg")

#     # Ghi ảnh tạm
#     cv2.imwrite(temp_path, frame)

#     # Đổi tên ảnh tạm thành file chính
#     shutil.move(temp_path, final_path)


# # Hàm tạo đường dẫn lưu ảnh theo cấu trúc year/month/day/hhmmss.jpg
# def get_detection_image_path():
#     now = datetime.now()
#     year_folder = os.path.join(BASE_DIR, str(now.year))
#     month_folder = os.path.join(year_folder, f"{now.month:02d}")
#     day_folder = os.path.join(month_folder, f"{now.day:02d}")

#     # Tạo thư mục nếu chưa tồn tại
#     os.makedirs(day_folder, exist_ok=True)

#     # Tạo file name theo định dạng hhmmss.jpg
#     timestamp = now.strftime("%H%M%S")
#     return os.path.join(day_folder, f"{timestamp}.jpg")


# def image_saver():
#     while True:
#         frame, save_path, temp_path, final_path = image_queue.get()
#         if frame is None:  # Tín hiệu để dừng luồng
#             break

#         try:
#             # Lưu ảnh vào thư mục year/month/day
#             cv2.imwrite(save_path, frame)

#             # Ghi ảnh tạm cho file image.jpg
#             cv2.imwrite(temp_path, frame)

#             # Đổi tên file tạm thành file chính
#             shutil.move(temp_path, final_path)

#         except PermissionError as e:
#             pass
#         except Exception as e:
#             pass
#         image_queue.task_done()


# # Khởi tạo luồng phụ
# thread = Thread(target=image_saver, daemon=True)
# thread.start()


# class Predictor(object):
#     def __init__(
#         self,
#         model,
#         exp,
#         cls_names=COCO_CLASSES,
#         trt_file=None,
#         decoder=None,
#         device="cpu",
#         fp16=False,
#         legacy=False,
#     ):
#         self.model = model
#         self.cls_names = cls_names
#         self.decoder = decoder
#         self.num_classes = exp.num_classes
#         self.confthre = exp.test_conf
#         self.nmsthre = exp.nmsthre
#         self.test_size = exp.test_size
#         self.device = device
#         self.fp16 = fp16
#         self.preproc = ValTransform(legacy=legacy)
#         if trt_file is not None:
#             from torch2trt import TRTModule

#             model_trt = TRTModule()
#             model_trt.load_state_dict(torch.load(trt_file))
#             x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
#             self.model(x)
#             self.model = model_trt

#     def inference(self, img):
#         img_info = {"id": 0}
#         img_info["raw_img"] = img
#         img_info["height"], img_info["width"] = img.shape[:2]

#         ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
#         img_info["ratio"] = ratio

#         img, _ = self.preproc(img, None, self.test_size)
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = img.float()
#         if self.device == "gpu":
#             img = img.cuda()
#             if self.fp16:
#                 img = img.half()

#         with torch.no_grad():
#             outputs = self.model(img)
#             if self.decoder is not None:
#                 outputs = self.decoder(outputs, dtype=outputs.type())
#             outputs = postprocess(
#                 outputs,
#                 self.num_classes,
#                 self.confthre,
#                 self.nmsthre,
#                 class_agnostic=True,
#             )
#         return outputs, img_info

#     def visual(self, output, img_info, cls_conf=0.35):
#         ratio = img_info["ratio"]
#         img = img_info["raw_img"]
#         if output is None:
#             return img
#         output = output.cpu()
#         bboxes = output[:, 0:4]
#         bboxes /= ratio
#         cls = output[:, 6]
#         scores = output[:, 4] * output[:, 5]
#         vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
#         return vis_res


# def main():
#     # Thiết lập các tham số
#     experiment_name = None
#     model_name = None
#     exp_file = "/YOLOX/exps/example/custom/yolox_s.py"
#     ckpt_file = "D:/Honda_PlusVN/Python/Python_Project/YOLOX/80.pth"
#     device = "gpu"
#     conf = 0.5
#     nms = 0.3
#     tsize = None
#     fp16 = False
#     legacy = False
#     fuse = False
#     trt = True  # Sử dụng TensorRT

#     # Lấy thiết lập từ file mô tả
#     exp = get_exp(exp_file, model_name)
#     if experiment_name is None:
#         experiment_name = exp.exp_name

#     file_name = os.path.join(exp.output_dir, experiment_name)
#     os.makedirs(file_name, exist_ok=True)

#     logger.info(f"Running inference on device: {device}")
#     logger.info(f"Using TensorRT: {trt}")

#     exp.test_conf = conf
#     exp.nmsthre = nms
#     if tsize is not None:
#         exp.test_size = (tsize, tsize)

#     model = exp.get_model()
#     logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

#     if device == "gpu":
#         model.cuda()
#         if fp16:
#             model.half()
#     model.eval()

#     if not trt:
#         if ckpt_file is None:
#             ckpt_file = os.path.join(file_name, "best_ckpt.pth")
#         logger.info("loading checkpoint")
#         ckpt = torch.load(ckpt_file, map_location="cpu")
#         model.load_state_dict(ckpt["model"])
#         logger.info("loaded checkpoint done.")

#     if fuse:
#         logger.info("\tFusing model...")
#         model = fuse_model(model)

#     if trt:
#         assert not fuse, "TensorRT model does not support model fusing!"

#         trt_file = os.path.join(file_name, "model_trt.pth")
#         assert os.path.exists(
#             trt_file
#         ), "TensorRT model not found! Run conversion script first."
#         model.head.decode_in_inference = False
#         decoder = model.head.decode_outputs
#         logger.info("Using TensorRT for inference")

#     else:
#         trt_file = None
#         decoder = None

#     predictor = Predictor(
#         model, exp, COCO_CLASSES, trt_file, decoder, device, fp16, legacy
#     )

#     pTime = 0

#     while True:
#         ret, frame = laptop_camera.read()

#         if ret:
#             resized_frame = cv2.resize(frame, (800, 800))

#             # Tính toán FPS
#             curr_time = time.time()
#             fps = 1 / (curr_time - pTime)
#             pTime = curr_time

#             try:
#                 outputs, img_info = predictor.inference(resized_frame)
#                 annotated_frame = predictor.visual(
#                     outputs[0], img_info, predictor.confthre
#                 )

#                 # Hiển thị FPS
#                 cv2.putText(
#                     annotated_frame,
#                     f"FPS: {int(fps)}",
#                     (0, 50),
#                     cv2.FONT_HERSHEY_PLAIN,
#                     3,
#                     (255, 255, 0),
#                     3,
#                 )

#                 # Nếu phát hiện đối tượng
#                 if outputs is not None and len(outputs[0]) > 0:
#                     send_command_to_arduino("1")  # Gửi tín hiệu bật còi đến Arduino
#                     save_path = get_detection_image_path()
#                     temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
#                     final_path = os.path.join(BASE_DIR, "image.jpg")

#                     # Đẩy ảnh và thông tin vào hàng đợi
#                     image_queue.put((resized_frame, save_path, temp_path, final_path))
#                 else:
#                     send_command_to_arduino("0")  # Gửi tín hiệu tắt còi đến Arduino

#                 # Hiển thị ảnh
#                 cv2.imshow("YOLOX Inference Laptop Camera", annotated_frame)

#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#             except Exception as e:
#                 print(e)

#     laptop_camera.release()
#     cv2.destroyAllWindows()

#     # Dừng luồng phụ
#     image_queue.put((None, None, None, None))
#     thread.join()


# if __name__ == "__main__":
#     main()


import os
import cv2
import time
import torch
from datetime import datetime
from threading import Thread
from queue import Queue
from loguru import logger
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import shutil


import signal
import atexit


# Hàm gửi tín hiệu tắt còi
def turn_off_buzzer():
    try:
        arduino.write(b"0")  # Gửi tín hiệu '0' để tắt còi
        print("Buzzer turned off.")
    except Exception as e:
        print(f"Failed to turn off buzzer: {e}")


# Gắn hàm tắt còi vào sự kiện kết thúc chương trình
atexit.register(turn_off_buzzer)


# Bắt tín hiệu dừng chương trình (Ctrl+C hoặc đóng chương trình)
def signal_handler(sig, frame):
    print("Program is stopping...")
    turn_off_buzzer()
    arduino.close()  # Đóng kết nối với Arduino
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
# Khởi tạo camera laptop
laptop_camera = cv2.VideoCapture(0)

# Thư mục gốc để lưu ảnh
BASE_DIR = "Thelf_Detection"

# Hàng đợi để lưu ảnh
image_queue = Queue()

import os
import shutil
import cv2

BASE_DIR = "Thelf_Detection"

import serial
import serial.tools.list_ports

# Tìm cổng COM đúng
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"Found port: {port.device}")

try:
    # Mở cổng COM (thay COM8 bằng cổng đúng)
    arduino = serial.Serial(port="COM7", baudrate=9600, timeout=1)
    print("Arduino connected successfully!")
except serial.SerialException as e:
    print(f"Failed to connect: {e}")


def send_command_to_arduino(command):
    try:
        arduino.write(command.encode())  # Gửi tín hiệu đến Arduino
    except Exception as e:
        print(f"Error sending command to Arduino: {e}")


def save_image(frame):
    temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
    final_path = os.path.join(BASE_DIR, "image.jpg")

    # Ghi ảnh tạm
    cv2.imwrite(temp_path, frame)

    # Đổi tên ảnh tạm thành file chính
    shutil.move(temp_path, final_path)


# Hàm tạo đường dẫn lưu ảnh theo cấu trúc year/month/day/hhmmss.jpg
def get_detection_image_path():
    now = datetime.now()
    year_folder = os.path.join(BASE_DIR, str(now.year))
    month_folder = os.path.join(year_folder, f"{now.month:02d}")
    day_folder = os.path.join(month_folder, f"{now.day:02d}")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(day_folder, exist_ok=True)

    # Tạo file name theo định dạng hhmmss.jpg
    timestamp = now.strftime("%H%M%S")
    return os.path.join(day_folder, f"{timestamp}.jpg")


def image_saver():
    while True:
        frame, save_path, temp_path, final_path = image_queue.get()
        if frame is None:  # Tín hiệu để dừng luồng
            break

        try:
            # Lưu ảnh vào thư mục year/month/day
            cv2.imwrite(save_path, frame)

            # Ghi ảnh tạm cho file image.jpg
            cv2.imwrite(temp_path, frame)

            # Đổi tên file tạm thành file chính
            shutil.move(temp_path, final_path)

        except PermissionError as e:
            pass
        except Exception as e:
            pass
        image_queue.task_done()


# Khởi tạo luồng phụ
thread = Thread(target=image_saver, daemon=True)
thread.start()


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
        img_info["raw_img"] = img
        img_info["height"], img_info["width"] = img.shape[:2]

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
    # Thiết lập các tham số
    experiment_name = None
    model_name = None
    exp_file = "/YOLOX/exps/example/custom/yolox_s.py"
    ckpt_file = "/YOLOX/80.pth"
    device = "gpu"
    conf = 0.5
    nms = 0.3
    tsize = None
    fp16 = False
    legacy = False
    fuse = False
    trt = True  # Sử dụng TensorRT

    # Lấy thiết lập từ file mô tả
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
        ret, frame = laptop_camera.read()

        if ret:
            resized_frame = cv2.resize(frame, (1024, 1024))

            # Tính toán FPS
            curr_time = time.time()
            fps = 1 / (curr_time - pTime)
            pTime = curr_time

            try:
                outputs, img_info = predictor.inference(resized_frame)
                annotated_frame = predictor.visual(
                    outputs[0], img_info, predictor.confthre
                )

                # Hiển thị FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {int(fps)}",
                    (0, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 255, 0),
                    3,
                )

                # Nếu phát hiện đối tượng
                if outputs is not None and len(outputs[0]) > 0:
                    send_command_to_arduino("1")  # Gửi tín hiệu bật còi đến Arduino
                    save_path = get_detection_image_path()
                    temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
                    final_path = os.path.join(BASE_DIR, "image.jpg")

                    # Đẩy ảnh và thông tin vào hàng đợi
                    image_queue.put((resized_frame, save_path, temp_path, final_path))
                else:
                    send_command_to_arduino("0")  # Gửi tín hiệu tắt còi đến Arduino

                # Hiển thị ảnh
                cv2.imshow("YOLOX Inference Laptop Camera", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                print(e)

    laptop_camera.release()
    cv2.destroyAllWindows()

    # Dừng luồng phụ
    image_queue.put((None, None, None, None))
    thread.join()


if __name__ == "__main__":
    main()
