import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

class ROIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video ROI Selector")

        # State
        self.video_path = None
        self.frame = None
        self.roi = None
        self._start = None
        self.rect = None
        self.tkimg = None

        # UI: Load button
        load_btn = tk.Button(self, text="Load Video", command=self.load_video)
        load_btn.pack(fill="x", pady=5)

        # UI: Canvas for showing frame + drawing ROI
        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack()

        # UI: Confirm / Start buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", pady=5)
        tk.Button(btn_frame, text="Confirm ROI", command=self.confirm_roi).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Start Processing", command=self.start_processing).pack(side="left", expand=True, fill="x")

        # Bind mouse events for ROI drawing
        self.canvas.bind("<ButtonPress-1>",     self.on_button_press)
        self.canvas.bind("<B1-Motion>",         self.on_move)
        self.canvas.bind("<ButtonRelease-1>",   self.on_button_release)

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files","*.mp4"),("All files","*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Could not read the selected video.")
            return

        self.video_path = path
        self.frame = frame
        self.display_frame(frame)

    def display_frame(self, frame):
        """Convert and show an OpenCV BGR frame on the canvas."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

    def on_button_press(self, event):
        if self.frame is None:
            return
        self._start = (event.x, event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")

    def on_move(self, event):
        if self.rect and self._start:
            x0, y0 = self._start
            self.canvas.coords(self.rect, x0, y0, event.x, event.y)

    def on_button_release(self, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = event.x, event.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self.roi = (x, y, w, h)

    def confirm_roi(self):
        if self.roi:
            print("ROI coords (x, y, w, h):", self.roi)
        else:
            messagebox.showwarning("Warning", "Please draw an ROI first.")

    def start_processing(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Load a video first.")
            return
        if not self.roi:
            messagebox.showwarning("Warning", "Draw an ROI first.")
            return

        # HERE: hook your YOLO tracking/counting logic,
        # passing self.video_path and self.roi into it.
        print("Starting processing on:", self.video_path)
        print("Using ROI:", self.roi)

        self.destroy()  # close the GUI

if __name__ == "__main__":
    app = ROIApp()
    app.mainloop()
