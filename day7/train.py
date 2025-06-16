from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    # model = YOLO(r"C:\Users\hyxhyx\Desktop\Practice\project\YOLO\ultralytics\runs\detect\train\weights\best.pt")
    # model.train(data = "coco8.yaml",imgsz=640,epochs=20,batch=16)

    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    # model.train(data = r"C:\Users\hyxhyx\Desktop\Practice\project\YOLO\ultralytics\TrafficSignDetection.v9\TrafficSignDetection.v9.yaml",imgsz=640,epochs=20,batch=16)

    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    # model.train(data="coco8.yaml", imgsz=640, epochs=20, batch=16)

    # model = YOLO("yolo12n.yaml").load("yolo12n.pt")
    # model.train(data="coco8.yaml",imgsz=640, epochs=20, batch=16)

    # model = YOLO("yolo12n.yaml").load("yolo12n.pt")
    # model.train(data="data.yaml", imgsz=640, epochs=20, batch=16)

    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    model.train(data="data.yaml", imgsz=640, epochs=20, batch=16)

    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")
    # model.train(data="data.yaml", imgsz=640, epochs=20, batch=16)