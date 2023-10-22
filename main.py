import cv2 as cv
import numpy as np

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

image_width = 600
image_height = 600

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

threshold = 0.2

# Capture a frame from the external camera
cap = cv.VideoCapture("USB\VID_046D&PID_0819&REV_0010&MI_00") # Replace this with the device ID of your external camera
ret, img = cap.read()

# If the frame was captured successfully, process it
if ret:
    photo_height = img.shape[0]
    photo_width = img.shape[1]

    # Resize the image to the input size of the model
    img = cv.resize(img, (image_width, image_height))

    # Create a blob from the image
    blob = cv.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass the network
    out = net.forward()

    # Get the output of the network
    out = out[:, :19, :, :]

    # Assert that the number of body parts is equal to the number of output channels
    assert(len(BODY_PARTS) == out.shape[1])

    # Create a list of points
    points = []

    # Iterate over the body parts
    for i in range(len(BODY_PARTS)):
        # Slice the heatmap of the corresponding body part
        heatmap = out[0, i, :, :]

        # Find the global maximum of the heatmap
        _, conf, _, point = cv.minMaxLoc(heatmap)

        # Convert the point coordinates to the original image size
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]

        # Add the point to the list if its confidence is higher than the threshold
        points.append((int(x), int(y)) if conf > threshold else None)

    # Draw the pose skeleton on the image
    