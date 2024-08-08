import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8l.pt")


def process_frame(frame,model):

  results = model.track(frame)
  cls_person = []
  target_class = [2,3,4,5,6,7]

  for box in results[0].boxes:
      box_class = box.cls.item()
      for i in target_class:
          if i==box_class:
             cls_person.append(box)
          
  coordinates_cls_2 = [box.xyxy for box in cls_person]

  for box in coordinates_cls_2:
      x, y, x1, y1 = map(int, box[0])
      cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(frame, "Car", (x + 2, y - 10), font, 0.6, (255, 0, 0), 2)
  return frame
video_path = "4K Video of Highway Traffic!.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

out = cv2.VideoWriter('Car_datect.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, size)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        processed_frame = process_frame(frame,model)
        cv2.imshow("YOLOv8 Tracking", processed_frame)
     
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

out.release()
cap.release()
cv2.destroyAllWindows()
