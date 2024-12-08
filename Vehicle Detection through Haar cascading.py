import cv2

# Load the cascade
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Make sure the XML file is in the same directory

# Load the video or image
cap = cv2.VideoCapture('test_video.mp4')  # Replace with your video path or use 0 for webcam

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop if no frame is captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the image with adjusted parameters
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected vehicles
    for (x, y, w, h) in cars:
        # Optional: Filtering based on width and height to reduce false positives
        if w > 40 and h > 40:  # Adjust these values based on your specific needs
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
