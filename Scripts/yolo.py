#import all the required libraries

from ultralytics import YOLO

#Load the yolo model
model = YOLO("D:\Football Assignment\custom_model\last.pt")

#object detection
results = model.predict(source=r"D:\Football Assignment\Input videos\15sec_input_720p.mp4", save=True)

#Tracker detection

#results = model.track(source=r"D:\Football Assignment\Input videos\15sec_input_720p.mp4", save=True, persist=True)