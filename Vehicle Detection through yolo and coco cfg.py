import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load the COCO class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load an image
image = cv2.imread('test_image.jpg')  # Replace with your image path

if image is None:
    print("Error: Could not read the image. Check the file path.")
    exit()

# Resize the image to fit your display if necessary
image = cv2.resize(image, (800, 600))  # Resize as needed

# Preprocess the image
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Perform the forward pass to get outputs
outputs = net.forward(net.getUnconnectedOutLayersNames())

# Initialize lists to hold detection data
class_ids = []
confidences = []
boxes = []

# Process the outputs to get bounding boxes and confidences
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Debugging: print class id and confidence
        print(f"Class ID: {class_id}, Confidence: {confidence}")

        if confidence > 0.2:  # Set a confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Get the box coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Draw bounding boxes for detected objects
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = (0, 255, 0)  # Green color for bounding boxes
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show the resulting image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 800, 600)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
