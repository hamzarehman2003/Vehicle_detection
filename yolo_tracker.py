# tracker.py

import cv2, time, numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import defaultdict
# for R-CNN
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from scipy.spatial import distance as dist
import statistics

# ——————————————————————————————————————————————————————
# CentroidTracker for R-CNN branch
# ——————————————————————————————————————————————————————
class CentroidTracker:
    def __init__(self, maxDisappeared=10, maxDistance=50):
        self.nextObjectID   = 0
        self.objects        = {}   # id -> centroid
        self.boxes          = {}   # id -> bbox
        self.labels         = {}   # id -> class name
        self.disappeared    = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance    = maxDistance

    def register(self, centroid, box, label):
        oid = self.nextObjectID
        self.objects[oid]     = centroid
        self.boxes[oid]       = box
        self.labels[oid]      = label
        self.disappeared[oid] = 0
        self.nextObjectID    += 1

    def deregister(self, oid):
        for d in (self.objects, self.boxes, self.labels, self.disappeared):
            del d[oid]

    def update(self, rects, rect_labels):
        # rects: list of [x1,y1,x2,y2], rect_labels: list of names
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects, self.boxes, self.labels

        inputCentroids = np.zeros((len(rects),2), dtype="int")
        for i, (x1,y1,x2,y2) in enumerate(rects):
            inputCentroids[i] = (int((x1+x2)/2), int((y1+y2)/2))

        if not self.objects:
            for i, c in enumerate(inputCentroids):
                self.register(c, rects[i], rect_labels[i])
        else:
            oids = list(self.objects)
            ocs  = list(self.objects.values())
            D    = dist.cdist(np.array(ocs), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (r,c) in zip(rows, cols):
                if r in usedRows or c in usedCols or D[r,c] > self.maxDistance:
                    continue
                oid = oids[r]
                self.objects[oid]     = inputCentroids[c]
                self.boxes[oid]       = rects[c]
                self.labels[oid]      = rect_labels[c]
                self.disappeared[oid] = 0
                usedRows.add(r); usedCols.add(c)

            unusedR = set(range(D.shape[0])) - usedRows
            for r in unusedR:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

            unusedC = set(range(D.shape[1])) - usedCols
            for c in unusedC:
                self.register(inputCentroids[c], rects[c], rect_labels[c])

        return self.objects, self.boxes, self.labels

# COCO label names for R-CNN
COCO_NAMES = [
    '__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
    'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

# ——————————————————————————————————————————————————————
# Main GUI: ROI + model choice + start
# ——————————————————————————————————————————————————————
class ROIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video ROI & Model Selector")

        # state
        self.video_path = None
        self.frame      = None
        self.roi        = None
        self._start     = None
        self.rect       = None
        self.tkimg      = None

        # --- model choice ---
        mfrm = tk.Frame(self)
        mfrm.pack(fill="x", pady=5)
        tk.Label(mfrm, text="Choose model:").pack(side="left")
        self.model_var = tk.StringVar(value="YOLO")
        tk.Radiobutton(mfrm, text="YOLOv8",   variable=self.model_var, value="YOLO").pack(side="left")
        tk.Radiobutton(mfrm, text="Faster R-CNN", variable=self.model_var, value="RCNN").pack(side="left")

        # --- load video button ---
        tk.Button(self, text="Load Video", command=self.load_video)\
          .pack(fill="x", pady=5)

        # --- canvas for ROI drawing ---
        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack()

        # --- confirm ROI / start ---
        btns = tk.Frame(self)
        btns.pack(fill="x", pady=5)
        tk.Button(btns, text="Confirm ROI",      command=self.confirm_roi).pack(side="left", expand=True, fill="x")
        tk.Button(btns, text="Start Processing", command=self.start_processing).pack(side="left", expand=True, fill="x")

        # mouse bindings
        self.canvas.bind("<ButtonPress-1>",    self.on_press)
        self.canvas.bind("<B1-Motion>",        self.on_move)
        self.canvas.bind("<ButtonRelease-1>",  self.on_release)

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("MP4","*.mp4"),("All","*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error","Could not read video.")
            return
        self.video_path = path
        self.frame      = frame
        self._draw(frame)

    def _draw(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def on_press(self, e):
        if self.frame is None: return
        self._start = (e.x,e.y)
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(e.x,e.y,e.x,e.y, outline="red")

    def on_move(self, e):
        if not self._start: return
        x0,y0 = self._start
        self.canvas.coords(self.rect, x0,y0, e.x,e.y)

    def on_release(self, e):
        if not self._start: return
        x0,y0 = self._start; x1,y1 = e.x,e.y
        x,y = min(x0,x1), min(y0,y1)
        w,h = abs(x1-x0), abs(y1-y0)
        self.roi = (x,y,w,h)

    def confirm_roi(self):
        if self.roi:
            messagebox.showinfo("ROI", f"{self.roi}")
        else:
            messagebox.showwarning("Draw ROI first", "")

    def start_processing(self):
        if not self.video_path: messagebox.showwarning("Load a video", ""); return
        if not self.roi:        messagebox.showwarning("Draw ROI first",""); return
        choice = self.model_var.get()
        self.destroy()
        metrics = process_video(self.video_path, self.roi, choice)
        show_results(metrics)

# ——————————————————————————————————————————————————————
# Processing: YOLO vs. R-CNN branch
# ——————————————————————————————————————————————————————
def process_video(video_path, roi, model_type):
    # common setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    x_roi,y_roi,w_roi,h_roi = roi

    # data
    target = {"person","car","truck","bus","motorcycle","bicycle"}
    counts = defaultdict(int)
    centroids = defaultdict(list)
    counted_ids = set()

    # performance
    frame_times = []
    frame_count = 0
    t0 = time.time()

    if model_type == "YOLO":
        model = YOLO("yolov8n.pt")
        stream = model.track(source=video_path, tracker="bytetrack.yaml", stream=True, verbose=False)
        for result in stream:
            frame = result.orig_img
            frame_count += 1
            f0 = time.time()

            # ROI box
            cv2.rectangle(frame,(x_roi,y_roi),(x_roi+w_roi,y_roi+h_roi),(255,255,0),2)

            # each tracked box
            for box in result.boxes:
                cls = int(box.cls[0]); oid = int(box.id[0])
                name = model.names[cls]
                if name not in target: continue
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                if not (x_roi<cx<x_roi+w_roi and y_roi<cy<y_roi+h_roi): continue

                # unique count
                if oid not in counted_ids:
                    counted_ids.add(oid)
                    counts[name]+=1

                # speed
                centroids[oid].append((cx,cy))
                speed=0
                pts = centroids[oid]
                if len(pts)>1:
                    dx,dy = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
                    pix = np.hypot(dx,dy)
                    ft  = pix/8
                    km  = ft * 0.0003048
                    sec = 1/fps
                    speed = km/(sec/3600)

                col = (0,0,255) if speed>30 else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"ID:{oid}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                cv2.putText(frame,f"{speed:.1f} km/h",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

            cv2.putText(frame,f"Counts: {dict(counts)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1)==ord('q'): break

            f1=time.time(); frame_times.append(f1-f0)

    else:  # R-CNN branch
        # load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rcnn   = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        tf     = T.Compose([T.ToTensor()])
        tracker= CentroidTracker()

        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            f0 = time.time()

            # ROI box
            cv2.rectangle(frame,(x_roi,y_roi),(x_roi+w_roi,y_roi+h_roi),(255,255,0),2)

            # R-CNN detection
            img = tf(frame).to(device)
            with torch.no_grad():
                out = rcnn([img])[0]
            boxes  = out['boxes'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            scores = out['scores'].cpu().numpy()

            # filter
            rects, rect_labels = [], []
            for b,lab,s in zip(boxes,labels,scores):
                name = COCO_NAMES[lab]
                if name in target and s>0.5:
                    x1,y1,x2,y2 = b.astype(int)
                    cx,cy = (x1+x2)//2,(y1+y2)//2
                    if x_roi<cx<x_roi+w_roi and y_roi<cy<y_roi+h_roi:
                        rects.append([x1,y1,x2,y2])
                        rect_labels.append(name)

            # update tracker
            objs, bxs, labs = tracker.update(rects, rect_labels)
            # draw & count
            for oid, cen in objs.items():
                box = bxs[oid]; name = labs[oid]
                if oid not in counted_ids:
                    counted_ids.add(oid)
                    counts[name]+=1

                x1,y1,x2,y2 = box
                cx,cy = cen
                centroids[oid].append(cen)
                speed=0
                pts=centroids[oid]
                if len(pts)>1:
                    dx,dy=np.hypot(pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]),0
                    pix = np.hypot(pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1])
                    ft  = pix/8
                    km  = ft*0.0003048
                    sec = 1/fps
                    speed = km/(sec/3600)

                col = (0,0,255) if speed>30 else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"ID:{oid}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                cv2.putText(frame,f"{speed:.1f} km/h",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

            cv2.putText(frame,f"Counts: {dict(counts)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1)==ord('q'): break

            f1=time.time(); frame_times.append(f1-f0)
        cap.release()

    cv2.destroyAllWindows()

    t1 = time.time()
    total = t1 - t0
    return {
        "model":      model_type,
        "frames":     frame_count,
        "total_time": total,
        "avg_fps":    frame_count/total if total>0 else 0,
        "min_ft":     min(frame_times)*1000 if frame_times else 0,
        "max_ft":     max(frame_times)*1000 if frame_times else 0,
        "avg_ft":     statistics.mean(frame_times)*1000 if frame_times else 0,
        "med_ft":     statistics.median(frame_times)*1000 if frame_times else 0,
        "counts":     dict(counts)
    }

# ——————————————————————————————————————————————————————
# Final GUI: show all metrics
# ——————————————————————————————————————————————————————
def show_results(m):
    w = tk.Tk()
    w.title(f"Results ({m['model']})")

    tk.Label(w, text=f"Frames processed       : {m['frames']}").pack(anchor="w")
    tk.Label(w, text=f"Total processing time  : {m['total_time']:.2f}s").pack(anchor="w")
    tk.Label(w, text=f"Average FPS            : {m['avg_fps']:.1f}").pack(anchor="w")
    tk.Label(w, text=f"Min frame time         : {m['min_ft']:.1f} ms").pack(anchor="w")
    tk.Label(w, text=f"Max frame time         : {m['max_ft']:.1f} ms").pack(anchor="w")
    tk.Label(w, text=f"Avg frame time         : {m['avg_ft']:.1f} ms").pack(anchor="w")
    tk.Label(w, text=f"Median frame time      : {m['med_ft']:.1f} ms").pack(anchor="w")
    tk.Label(w, text=f"Detected counts        : {m['counts']}").pack(anchor="w")

    w.mainloop()

if __name__ == "__main__":
    app = ROIApp()
    app.mainloop()
