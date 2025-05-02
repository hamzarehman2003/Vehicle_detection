# tracker.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# --- GUI for ROI selection and start button ---
class ROIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video ROI Selector")

        # state
        self.video_path = None
        self.frame      = None
        self.roi        = None
        self._start     = None
        self.rect       = None
        self.tkimg      = None

        # Load button
        tk.Button(self, text="Load Video", command=self.load_video)\
          .pack(fill="x", pady=5)

        # Canvas for first frame + drawing
        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack()

        # Confirm & Start
        frm = tk.Frame(self)
        frm.pack(fill="x", pady=5)
        tk.Button(frm, text="Confirm ROI",      command=self.confirm_roi).pack(side="left", expand=True, fill="x")
        tk.Button(frm, text="Start Processing", command=self.start_processing).pack(side="left", expand=True, fill="x")

        # Bind mouse for drawing
        self.canvas.bind("<ButtonPress-1>",    self.on_press)
        self.canvas.bind("<B1-Motion>",        self.on_move)
        self.canvas.bind("<ButtonRelease-1>",  self.on_release)

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("MP4","*.mp4"),("All","*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Cannot read video.")
            return

        self.video_path = path
        self.frame      = frame
        self._draw_frame(frame)

    def _draw_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def on_press(self, event):
        if self.frame is None:
            return
        self._start = (event.x, event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")

    def on_move(self, event):
        if not self._start: return
        x0,y0 = self._start
        self.canvas.coords(self.rect, x0, y0, event.x, event.y)

    def on_release(self, event):
        if not self._start: return
        x0,y0 = self._start
        x1,y1 = event.x, event.y
        x,y = min(x0,x1), min(y0,y1)
        w,h = abs(x1-x0), abs(y1-y0)
        self.roi = (x, y, w, h)

    def confirm_roi(self):
        if self.roi:
            print("ROI coords:", self.roi)
        else:
            messagebox.showwarning("Warning", "Draw an ROI first.")

    def start_processing(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Load a video first.")
            return
        if not self.roi:
            messagebox.showwarning("Warning", "Draw an ROI first.")
            return
        # close GUI and begin tracking
        self.destroy()
        process_video(self.video_path, self.roi)

# --- Tracking / Counting / Speed Estimation ---
def process_video(video_path, roi):
    # load YOLO and data structures
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    target_classes = {"person","car","truck","bus","motorcycle","bicycle"}
    counted_ids    = set()
    centroids      = defaultdict(list)
    counts         = defaultdict(int)

    x_roi,y_roi,w_roi,h_roi = roi

    # stream frames through ByteTrack
    for result in model.track(source=video_path,
                              tracker="bytetrack.yaml",
                              stream=True,
                              verbose=False):

        frame = result.orig_img
        # draw the ROI box
        cv2.rectangle(frame,
                      (x_roi,y_roi),
                      (x_roi+w_roi,y_roi+h_roi),
                      (255,255,0), 2)

        # iterate each tracked box
        for box in result.boxes:
            cls  = int(box.cls[0])
            oid  = int(box.id[0])
            name = model.names[cls]
            if name not in target_classes:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy     = (x1+x2)//2, (y1+y2)//2

            # only inside ROI
            if not (x_roi<cx<x_roi+w_roi and y_roi<cy<y_roi+h_roi):
                continue

            # unique count
            if oid not in counted_ids:
                counted_ids.add(oid)
                counts[name] += 1

            # speed (8 px = 1 ft)
            centroids[oid].append((cx,cy))
            speed_kmh = 0.0
            pts = centroids[oid]
            if len(pts) >= 2:
                dx,dy  = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
                pix    = np.hypot(dx,dy)
                feet   = pix/8
                km     = feet * 0.0003048
                sec    = 1/fps
                speed_kmh = km / (sec/3600)

            color = (0,0,255) if speed_kmh>30 else (0,255,0)
            # draw box, ID, speed
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"ID:{oid}",      (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"{speed_kmh:.1f} km/h", (x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # overlay counts
        cv2.putText(frame, f"Counts: {dict(counts)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2)

        cv2.imshow("Tracking & Counting", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    # final report & accuracy
    print("Final counts:", dict(counts))
    try:
        actual = int(input("Enter actual number of cars: "))
        acc    = counts["car"] / actual * 100
        print(f"Car counting accuracy: {acc:.1f}%")
    except Exception:
        print("Skipping accuracy calculation.")

# --- Entry point ---
if __name__ == "__main__":
    app = ROIApp()
    app.mainloop()
