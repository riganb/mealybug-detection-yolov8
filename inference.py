from ultralytics import YOLO
import  cv2
import cvzone
import math

cap = cv2.VideoCapture('http://192.168.0.106:4747/video')
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('yolov8m_mealybug.pt')

classNames = ["mealyBug", "mealybug"]

while True:
    success, img  =cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = classNames[int(cls)]

            cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)



    cv2.imshow("Image", img)
    cv2.waitKey(1)