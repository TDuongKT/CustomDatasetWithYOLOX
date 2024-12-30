import cv2

# Open a connection to the default camera (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Camera connected. Press 'q' to quit.")

# Loop to read frames from the camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Display the frame in a window
    cv2.imshow("Laptop Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
