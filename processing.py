# processing.py

import cv2, time, numpy as np, statistics
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from scipy.spatial import distance as dist
from collections import defaultdict

# (Paste the CentroidTracker class from above here)

def run_model_on_video(input_path, output_path, model_type="YOLO"):
    target = {"person","car","truck","bus","motorcycle","bicycle"}

    # Video setup
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats
    counts      = defaultdict(int)
    centroids   = defaultdict(list)
    counted_ids = set()
    times       = []
    frames      = 0
    t0          = time.time()

    if model_type == "YOLO":
        model = YOLO("yolov8n.pt")
        stream = model.track(source=input_path, tracker="bytetrack.yaml", stream=True, verbose=False)
        for res in stream:
            frames += 1
            f0 = time.time()
            frame = res.orig_img

            for box in res.boxes:
                cls = int(box.cls[0]); oid = int(box.id[0])
                name = model.names[cls]
                if name not in target: continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2

                if oid not in counted_ids:
                    counted_ids.add(oid)
                    counts[name] += 1

                centroids[oid].append((cx,cy))
                speed = 0
                pts = centroids[oid]
                if len(pts)>1:
                    dx,dy = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
                    pix   = np.hypot(dx,dy)
                    feet  = pix/8
                    km    = feet*0.0003048
                    sec   = 1/fps
                    speed = km/(sec/3600)

                col = (0,0,255) if speed>30 else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"{speed:.1f} km/h",(x1,y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

            writer.write(frame)
            times.append(time.time()-f0)

    else:  # Faster R-CNN
        weights   = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        rcnn      = fasterrcnn_resnet50_fpn(weights=weights).to(
                       "cuda" if torch.cuda.is_available() else "cpu"
                   ).eval()
        categories= weights.meta["categories"]
        tf        = T.Compose([T.ToTensor()])
        tracker   = CentroidTracker()

        cap = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames   += 1
            f0        = time.time()

            img = tf(frame).to(rcnn.device)
            with torch.no_grad():
                out = rcnn([img])[0]

            rects, labs = [], []
            for b,lab,sc in zip(out["boxes"].cpu().numpy(),
                               out["labels"].cpu().numpy(),
                               out["scores"].cpu().numpy()):
                name = categories[lab]
                if sc<0.5 or name not in target: continue
                rects.append(b.astype(int))
                labs.append(name)

            objs, boxes, labels = tracker.update(rects, labs)
            for oid, cen in objs.items():
                name = labels[oid]
                if oid not in counted_ids:
                    counted_ids.add(oid)
                    counts[name] += 1

                x1,y1,x2,y2 = boxes[oid]
                centroids[oid].append(cen)
                speed = 0
                pts   = centroids[oid]
                if len(pts)>1:
                    dx = pts[-1][0]-pts[-2][0]
                    dy = pts[-1][1]-pts[-2][1]
                    pix = np.hypot(dx,dy)
                    feet= pix/8
                    km  = feet*0.0003048
                    sec = 1/fps
                    speed = km/(sec/3600)

                col = (0,0,255) if speed>30 else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"{speed:.1f} km/h",(x1,y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

            writer.write(frame)
            times.append(time.time()-f0)

        cap.release()

    writer.release()
    t1 = time.time()

    return {
        "model":      model_type,
        "frames":     frames,
        "total_time": t1-t0,
        "avg_fps":    frames/(t1-t0),
        "counts":     dict(counts)
    }
