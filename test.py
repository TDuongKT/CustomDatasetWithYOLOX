from pypylon import pylon

# Instantiate the pylon DeviceInfo object and use it to get the cameras
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) < 1:
    raise ValueError("Not enough cameras found")

# Create a camera object
camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

# Open the camera
camera.Open()
