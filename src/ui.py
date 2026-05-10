"""
ui.py
-----
Giao diện Tkinter chính cho ứng dụng nhận diện khẩu trang.
"""
 
from __future__ import annotations
 
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
 
BG_DARK   = "#0f1117"
BG_PANEL  = "#1a1d27"
BG_CARD   = "#22263a"
ACCENT    = "#00e5ff"
GREEN     = "#00e676"
RED       = "#ff1744"
TEXT_PRI  = "#e8eaf6"
TEXT_SEC  = "#7986cb"
BORDER    = "#2e3250"
 
VIDEO_W, VIDEO_H = 640, 480
 
 
class MaskDetectionUI:
    def __init__(self, root: tk.Tk, camera_factory):
        self.root = root
        self.camera_factory = camera_factory
        self.camera_thread = None
        self._latest_imgtk = None
 
        self.root.title("He Thong Nhan Dien Khau Trang")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
 
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
 
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK)
        header.pack(fill="x", padx=16, pady=(12, 4))
 
        tk.Label(header, text="MASK DETECTION",
                 font=("Courier", 20, "bold"), fg=ACCENT, bg=BG_DARK).pack(side="left")
 
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)
 
        # ── Body ──────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=4)
 
        # -- Video feed (trái) ----------------------------------------
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)
 
        self.video_label = tk.Label(left, bg="#000000",
                                    width=VIDEO_W, height=VIDEO_H)
        self.video_label.pack(padx=4, pady=4)
        self._show_placeholder()
 
        # -- Panel phải -----------------------------------------------
        right = tk.Frame(body, bg=BG_DARK, width=220)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
 
        # Stat cards
        self.fps_var    = tk.StringVar(value="--")
        self.total_var  = tk.StringVar(value="0")
        self.mask_var   = tk.StringVar(value="0")
        self.nomask_var = tk.StringVar(value="0")
        self.ratio_var  = tk.StringVar(value="0%")
 
        self._stat_row(right, "FPS",              self.fps_var,    ACCENT)
        self._stat_row(right, "Khuon mat",         self.total_var,  TEXT_PRI)
        self._stat_row(right, "Co khau trang",     self.mask_var,   GREEN)
        self._stat_row(right, "Khong deo",         self.nomask_var, RED)
        self._stat_row(right, "Ti le deo",         self.ratio_var,  GREEN)
 
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=8)
 
        # Nút BẮT ĐẦU
        self.start_btn = tk.Button(
            right, text="▶  BAT DAU",
            bg=ACCENT, fg="#000000",
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=8, pady=10,
            command=self._start_camera,
        )
        self.start_btn.pack(fill="x", pady=(0, 6))
 
        # Nút DỪNG
        self.stop_btn = tk.Button(
            right, text="⏹  DUNG",
            bg="#37474f", fg=TEXT_PRI,
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=8, pady=10,
            state="disabled",
            command=self._stop_camera,
        )
        self.stop_btn.pack(fill="x", pady=(0, 6))
 
        # Nút THOÁT
        tk.Button(
            right, text="✕  THOAT",
            bg="#b71c1c", fg="#ffffff",
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=8, pady=10,
            command=self._on_close,
        ).pack(fill="x")
 
        # ── Status bar ────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="San sang — Nhan BAT DAU de mo camera")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Helvetica", 9), fg=TEXT_SEC, bg=BG_DARK,
                 anchor="w").pack(fill="x", padx=16, pady=(4, 8))
 
    def _stat_row(self, parent, label, var, color):
        frame = tk.Frame(parent, bg=BG_CARD, padx=10, pady=6)
        frame.pack(fill="x", pady=2)
        tk.Label(frame, text=label, font=("Helvetica", 8),
                 fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
        tk.Label(frame, textvariable=var, font=("Courier", 22, "bold"),
                 fg=color, bg=BG_CARD).pack(anchor="w")
 
    def _show_placeholder(self):
        img = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        img[:] = (20, 20, 30)
        cv2.putText(img, "Camera chua mo", (VIDEO_W//2 - 130, VIDEO_H//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (60, 80, 100), 2)
        self._display_array(img)
 
    def _start_camera(self):
        if self.camera_thread and self.camera_thread.is_running:
            return
        try:
            self.camera_thread = self.camera_factory()
            self.camera_thread.on_frame = self._on_new_frame
            self.camera_thread.start()
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the khoi dong camera:\n{exc}")
            return
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Dang chay — Camera da mo")
 
    def _stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Da dung — Nhan BAT DAU de tiep tuc")
        self._show_placeholder()
        self._reset_stats()
 
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()
 
    def _on_new_frame(self, frame: np.ndarray, stats: dict):
        self.root.after(0, self._update_ui, frame, stats)
 
    def _update_ui(self, frame: np.ndarray, stats: dict):
        self._display_array(frame)
        self.fps_var.set(f"{stats['fps']:.0f}")
        self.total_var.set(str(stats["total"]))
        self.mask_var.set(str(stats["mask"]))
        self.nomask_var.set(str(stats["no_mask"]))
        total = stats["total"]
        ratio = int(stats["mask"] / total * 100) if total > 0 else 0
        self.ratio_var.set(f"{ratio}%")
        if total == 0:
            self.status_var.set("Khong phat hien khuon mat nao")
        elif stats["no_mask"] == 0:
            self.status_var.set(f"Tat ca {total} nguoi deu deo khau trang")
        else:
            self.status_var.set(f"{stats['no_mask']}/{total} nguoi KHONG deo khau trang")
 
    def _display_array(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = min(VIDEO_W / w, VIDEO_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        y0 = (VIDEO_H - nh) // 2
        x0 = (VIDEO_W - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(canvas)
        imgtk = ImageTk.PhotoImage(image=img)
        self._latest_imgtk = imgtk
        self.video_label.config(image=imgtk)
 
    def _reset_stats(self):
        self.fps_var.set("--")
        self.total_var.set("0")
        self.mask_var.set("0")
        self.nomask_var.set("0")
        self.ratio_var.set("0%")