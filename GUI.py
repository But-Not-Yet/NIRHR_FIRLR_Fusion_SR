import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time
import threading
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))


def is_rtsp_source(source):
    return isinstance(source, str) and source.lower().startswith("rtsp://")


class FusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI User Interface")
        self.root.geometry("900x600")

        self.power_on = False
        self.current_view = "Fusion"

        # ---- Live stream state ----
        self.cap = None
        self.streaming = False
        self._tk_img = None
        self.connect_timeout_job = None
        self.open_attempt_id = 0

        # ---- Low-latency reader state ----
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.reader_thread = None
        self.reader_running = False

        # ---- Recording state ----
        self.recording = False
        self.video_writer = None
        self.video_filename = None

        # ---- Stream sources ----
        self.stream_urls = {
            "NIR": "rtsp://192.168.137.37:8554/gs",
            "FIR": "rtsp://192.168.137.37:8554/thermal",
            "Fusion": "rtsp://192.168.137.37:8554/fusion",
            # For webcam testing:
            # "NIR": 0,
            # "FIR": 0,
            # "Fusion": 0,
        }

        # ================= Top Bar =================
        self.top_frame = tk.Frame(root, bg="#000000", height=60)
        self.top_frame.pack(fill="x")

        self.title_label = tk.Label(
            self.top_frame,
            text="FIR-NIR Fusion Control Panel",
            fg="white",
            bg="#000000",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(side="left", padx=20, pady=15)

        self.power_status = tk.Label(
            self.top_frame,
            text="Power: OFF",
            fg="red",
            bg="#000000",
            font=("Arial", 12, "bold")
        )
        self.power_status.pack(side="right", padx=20)

        # ============== Middle Section ==============
        self.middle_frame = tk.Frame(root)
        self.middle_frame.pack(fill="both", expand=True)

        self.control_frame = tk.Frame(self.middle_frame, width=250, bg="#ffffff")
        self.control_frame.pack(side="left", fill="y")

        self.screen = tk.Frame(self.middle_frame, bg="black")
        self.screen.pack(side="left", fill="both", expand=True)
        self.screen.pack_propagate(False)

        self.view_label = tk.Label(
            self.screen,
            text="View: Fusion",
            fg="lime",
            bg="black",
            font=("Arial", 14, "bold")
        )
        self.view_label.pack(anchor="nw", padx=10, pady=10)

        self.image_label = tk.Label(
            self.screen,
            bg="black",
            fg="white",
            text="(powered off)"
        )
        self.image_label.pack(fill="both", expand=True)

        # ========== Left Bar Buttons ==========
        tk.Button(self.control_frame, text="Power ON", width=20, command=self.cmd_power_on).pack(pady=2)
        tk.Button(self.control_frame, text="Power OFF", width=20, command=self.cmd_power_off).pack(pady=2)
        tk.Button(self.control_frame, text="Capture Photo", width=20, command=self.capture_photo).pack(pady=2)
        tk.Button(self.control_frame, text="Start Video", width=20, command=self.start_recording).pack(pady=2)
        tk.Button(self.control_frame, text="Stop Video", width=20, command=self.stop_recording).pack(pady=2)
        tk.Button(self.control_frame, text="Select View", width=20, command=self.select_view).pack(pady=20)

        # =========== Bottom Log Console ===========
        self.bottom_frame = tk.Frame(root, height=100)
        self.bottom_frame.pack(fill="x")

        self.log_console = scrolledtext.ScrolledText(self.bottom_frame, height=5, state="disabled")
        self.log_console.pack(fill="both", padx=10, pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._refresh_screen_text()

    # ---------- Logging ----------
    def log(self, msg: str, color: str = None):
        ts = time.strftime("%H:%M:%S")
        self.log_console.configure(state="normal")

        if color:
            if color not in self.log_console.tag_names():
                self.log_console.tag_config(color, foreground=color)
            self.log_console.insert("end", f"[{ts}] {msg}\n", color)
        else:
            self.log_console.insert("end", f"[{ts}] {msg}\n")

        self.log_console.see("end")
        self.log_console.configure(state="disabled")

    # ---------- UI helpers ----------
    def _update_power_ui(self):
        if self.power_on:
            self.power_status.config(text="Power: ON", fg="lime")
        else:
            self.power_status.config(text="Power: OFF", fg="red")

    def _refresh_screen_text(self):
        self.view_label.config(text=f"View: {self.current_view}")

    def _ensure_power(self):
        if not self.power_on:
            self.log("ERROR: Power is OFF — turn ON first.", color="red")
            return False
        return True

    def _get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    # ---------- Background reader ----------
    def _reader_loop(self):
        while self.reader_running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            with self.frame_lock:
                self.latest_frame = frame

    # ---------- Streaming ----------
    def _start_stream(self, source):
        self._stop_stream()

        self.image_label.config(image="", text="(connecting...)", fg="white", bg="black")
        self.streaming = False

        self.open_attempt_id += 1
        attempt_id = self.open_attempt_id

        def worker():
            cap = None
            first_frame = None
            error_msg = None

            try:
                if is_rtsp_source(source):
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    if cap is None or not cap.isOpened():
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                        cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source)

                if cap is None or not cap.isOpened():
                    error_msg = f"Failed to open stream: {source}"
                else:
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    ok, first_frame = cap.read()
                    if not ok or first_frame is None:
                        error_msg = f"Opened stream but no frame was received: {source}"

            except Exception as e:
                error_msg = f"Exception opening stream: {e}"

            self.root.after(
                0,
                lambda: self._finish_stream_open(attempt_id, source, cap, first_frame, error_msg)
            )

        threading.Thread(target=worker, daemon=True).start()

        if self.connect_timeout_job is not None:
            self.root.after_cancel(self.connect_timeout_job)

        self.connect_timeout_job = self.root.after(
            10000,
            lambda: self._stream_open_timeout(attempt_id, source)
        )

    def _stream_open_timeout(self, attempt_id, source):
        if attempt_id != self.open_attempt_id or self.streaming:
            return

        self.open_attempt_id += 1

        self.log(f"ERROR: Timed out after 5 seconds opening stream: {source}", color="red")
        self.image_label.config(image="", text="(stream timeout)", fg="white", bg="black")

        if self.connect_timeout_job is not None:
            self.root.after_cancel(self.connect_timeout_job)
            self.connect_timeout_job = None

    def _finish_stream_open(self, attempt_id, source, cap, first_frame, error_msg):
        if attempt_id != self.open_attempt_id:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return

        if self.connect_timeout_job is not None:
            self.root.after_cancel(self.connect_timeout_job)
            self.connect_timeout_job = None

        if error_msg:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.cap = None
            self.streaming = False
            self.log(f"ERROR: {error_msg}", color="red")
            self.image_label.config(image="", text="(stream failed)", fg="white", bg="black")
            return

        self.cap = cap
        self.streaming = True

        with self.frame_lock:
            self.latest_frame = first_frame

        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        self.log(f"Streaming started: {source}")
        self.image_label.config(text="")
        self._stream_loop()

    def _stop_stream(self):
        self.stop_recording(silent=True)
        self.streaming = False
        self.reader_running = False

        if self.connect_timeout_job is not None:
            self.root.after_cancel(self.connect_timeout_job)
            self.connect_timeout_job = None

        self.open_attempt_id += 1

        if self.reader_thread is not None and self.reader_thread.is_alive():
            try:
                self.reader_thread.join(timeout=0.2)
            except Exception:
                pass
        self.reader_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        with self.frame_lock:
            self.latest_frame = None

    def _stream_loop(self):
        if not self.streaming or self.cap is None:
            return

        frame = self._get_latest_frame()

        if frame is not None:
            if self.recording and self.video_writer is not None:
                try:
                    self.video_writer.write(frame)
                except Exception:
                    self.log("ERROR: Failed while writing video.", color="red")
                    self.stop_recording()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            w = max(1, self.screen.winfo_width() - 20)
            h = max(1, self.screen.winfo_height() - 60)

            if w > 10 and h > 10:
                frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)

            img = Image.fromarray(frame_rgb)
            self._tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self._tk_img, text="")
        else:
            self.image_label.config(image="", text="(waiting for frame)", fg="white", bg="black")

        self.root.after(15, self._stream_loop)

    # ---------- Button callbacks ----------
    def start_recording(self):
        if not self._ensure_power():
            return

        if self.cap is None or not self.streaming:
            self.log("ERROR: No live stream active — cannot record video.", color="red")
            return

        if self.recording:
            self.log("ERROR: Video recording is already in progress.", color="red")
            return

        frame = self._get_latest_frame()
        if frame is None:
            self.log("ERROR: Could not start recording — no frame received.", color="red")
            return

        height, width = frame.shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0 or fps > 120:
            fps = 20.0

        self.video_filename = f"{PATH}/{self.current_view.lower()}_video_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        self.video_writer = cv2.VideoWriter(
            self.video_filename,
            fourcc,
            fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            self.video_filename = None
            self.log("ERROR: Failed to start video recording.", color="red")
            return

        self.recording = True
        self.log(f"Started recording video: {self.video_filename}")

    def stop_recording(self, silent=False):
        was_recording = self.recording
        filename = self.video_filename

        self.recording = False

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        self.video_filename = None

        if not silent:
            if was_recording:
                self.log(f"Stopped recording video: {filename}")
            else:
                self.log("ERROR: No video recording is currently active.", color="red")

    def cmd_power_on(self):
        if self.power_on:
            return

        self.power_on = True
        self._update_power_ui()
        self.log("Power turned ON.")

        if self.current_view in self.stream_urls:
            src = self.stream_urls[self.current_view]
            self._start_stream(src)
        else:
            self.image_label.config(image="", text="(no stream configured)", fg="white", bg="black")

    def cmd_power_off(self):
        if not self.power_on:
            return

        self.power_on = False
        self._update_power_ui()
        self._stop_stream()
        self.image_label.config(image="", text="(powered off)", fg="white", bg="black")
        self.log("Power turned OFF.")

    def open_view(self, view):
        self.current_view = view
        self._refresh_screen_text()
        self.log(f"Switched view to {view}.")

        if not self.power_on:
            self.image_label.config(image="", text="(powered off)", fg="white", bg="black")
            return

        if view in self.stream_urls:
            src = self.stream_urls[view]
            self._start_stream(src)
        else:
            self._stop_stream()
            self.image_label.config(image="", text="(no stream configured)", fg="white", bg="black")

    def select_view(self):
        if not self._ensure_power():
            return

        new_window = tk.Toplevel(self.root)
        new_window.title("Select View")
        new_window.geometry("250x150")

        tk.Button(
            new_window,
            text="FIR View",
            width=15,
            command=lambda: (self.open_view("FIR"), new_window.destroy())
        ).pack(pady=2)

        tk.Button(
            new_window,
            text="NIR View",
            width=15,
            command=lambda: (self.open_view("NIR"), new_window.destroy())
        ).pack(pady=2)

        tk.Button(
            new_window,
            text="Fusion View",
            width=15,
            command=lambda: (self.open_view("Fusion"), new_window.destroy())
        ).pack(pady=2)

        tk.Button(new_window, text="Close", command=new_window.destroy).pack(pady=10)

    def capture_photo(self):
        if not self._ensure_power():
            return

        if self.cap is None or not self.streaming:
            self.log("ERROR: No live stream active — cannot capture.", color="red")
            return

        frame = self._get_latest_frame()
        if frame is None:
            self.log("ERROR: Capture failed — no frame received.", color="red")
            return

        filename = f"{PATH}/{self.current_view.lower()}_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        saved = cv2.imwrite(filename, frame)

        if saved:
            self.log(f"Saved snapshot: {filename}")
        else:
            self.log("ERROR: Failed to save snapshot. Check file permissions/location.", color="red")

    def on_close(self):
        try:
            self.cmd_power_off()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FusionGUI(root)
    root.mainloop()