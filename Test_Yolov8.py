import cv2
import json
import time

from ultralytics import YOLO
from pypylon import pylon


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

pTime = 0

model = YOLO("D:\\Downloads\\best_tray_7.pt")


while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():

        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        resized_img = img

        # Resize the image
        resized_img = cv2.resize(
            img, (1900, 1500)
        )  # Adjust the width and height as desiredPIR

        # checking time
        curr_time = time.time()
        fps = 1 / (
            curr_time - pTime
        )  # tính fps (Frames Per Second) - đây là chỉ số khung hình trên mỗi giây

        pTime = curr_time

        try:
            result = model.predict(
                source=resized_img,
                device=0,
                conf=0.25,
                iou=0.25,
                verbose=True,
                imgsz=(1280, 1280),
            )

            jsonResult = result[0].tojson()
            parsed_datas = json.loads(jsonResult)
            annotated_frame = result[0].plot()
            result_message = ""
            result_color = (255, 255, 0)
            total_NG = ""

            cv2.putText(
                annotated_frame,
                f"FPS: {int(fps)}",
                (0, 50),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 255, 0),
                3,
            )

            cv2.putText(
                annotated_frame,
                result_message,
                (0, 100),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                result_color,
                3,
            )

            annotated_frame = cv2.resize(
                annotated_frame, (1500, 800)
            )  # Adjust the width and height as desiredPIR
            cv2.imshow("Mieng de hang", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            print(e)

    else:
        print(grabResult.GrabSucceeded())
    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()
