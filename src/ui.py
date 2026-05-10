"""
ui.py
-----
Giao diện Tkinter chính cho ứng dụng nhận diện khẩu trang.
"""
 
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
 
 
# ── Màu sắc giao diện ──────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_PANEL  = "#1a1d27"
BG_CARD   = "#22263a"
ACCENT    = "#00e5ff"
GREEN     = "#00e676"
RED       = "#ff1744"
TEXT_PRI  = "#e8eaf6"
TEXT_SEC  = "#7986cb"
BORDER    = "#2e3250"
 
 
class MaskDetectionUI:
    """
    Giao diện Tkinter cho hệ thống nhận diện khẩu trang realtime.
 
    Parameters
    ----------
    camera_factory : callable() → CameraThread
        Hàm không tham số trả về một CameraThread mới (chưa start).
    """
 
    def __init__(self, root: tk.Tk, camera_factory):
        self.root = root
        self.camera_factory = camera_factory
        self.camera_thread = None
 
        # Trạng thái thống kê
        self._stats = {"total": 0, "mask": 0, "no_mask": 0, "fps": 0.0}
        self._latest_imgtk = None  # giữ ref tránh GC
 
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
 
    # ── Build UI ───────────────────────────────────────────────────────────────
 
    def _build_ui(self):
        self.root.title("Hệ Thống Nhận Diện Khẩu Trang")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)
 
        # ── Header ─────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK)
        header.pack(fill="x", padx=20, pady=(16, 0))
 
        tk.Label(
            header,
            text="MASK",
            font=("Courier", 26, "bold"),
            fg=ACCENT,
            bg=BG_DARK,
        ).pack(side="left")
        tk.Label(
            header,
            text=" DETECTION",
            font=("Courier", 26, "bold"),
            fg=TEXT_PRI,
            bg=BG_DARK,
        ).pack(side="left")
 
        tk.Label(
            header,
            text="Nhận diện khẩu trang realtime",
            font=("Helvetica", 10),
            fg=TEXT_SEC,
            bg=BG_DARK,
        ).pack(side="right", padx=4)
 
        # Đường kẻ ngang
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)
 
        # ── Thân chính ─────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=0)
 
        # -- Cột trái: video feed -----------------------------------------------
        left = tk.Frame(body, bg=BG_PANEL, bd=0, relief="flat")
        left.pack(side="left", fill="both", expand=True)
 
        cam_border = tk.Frame(left, bg=BORDER, padx=2, pady=2)
        cam_border.pack(padx=12, pady=12)
 
        self.video_label = tk.Label(
            cam_border,
            bg="#000000",
            cursor="crosshair",
        )
        self.video_label.pack()
        self._show_placeholder()
 
        # -- Cột phải: thống kê + điều khiển ------------------------------------
        right = tk.Frame(body, bg=BG_DARK, width=240)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
 
        # FPS indicator
        fps_card = self._make_card(right)
        fps_card.pack(fill="x", pady=(0, 10))
        tk.Label(fps_card, text="FPS", font=("Courier", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
        self.fps_var = tk.StringVar(value="—")
        tk.Label(fps_card, textvariable=self.fps_var, font=("Courier", 28, "bold"),
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
 
        # Stat cards
        self.total_var   = tk.StringVar(value="0")
        self.mask_var    = tk.StringVar(value="0")
        self.nomask_var  = tk.StringVar(value="0")
 
        self._stat_card(right, "👥  KHUÔN MẶT",  self.total_var,  TEXT_PRI).pack(fill="x", pady=(0, 8))
        self._stat_card(right, "✅  ĐEO KHẨU TRANG", self.mask_var, GREEN).pack(fill="x", pady=(0, 8))
        self._stat_card(right, "❌  KHÔNG ĐEO",   self.nomask_var, RED).pack(fill="x", pady=(0, 16))
 
        # Progress bar tỷ lệ
        pb_card = self._make_card(right)
        pb_card.pack(fill="x", pady=(0, 16))
        tk.Label(pb_card, text="Tỷ lệ đeo khẩu trang", font=("Helvetica", 9),
                 fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
 
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Mask.Horizontal.TProgressbar",
            troughcolor=BG_DARK, background=GREEN,
            thickness=14, borderwidth=0,
        )
        self.progress = ttk.Progressbar(
            pb_card, style="Mask.Horizontal.TProgressbar",
            orient="horizontal", length=200, mode="determinate",
        )
        self.progress.pack(fill="x", pady=(6, 2))
        self.ratio_var = tk.StringVar(value="0 %")
        tk.Label(pb_card, textvariable=self.ratio_var, font=("Courier", 11, "bold"),
                 fg=GREEN, bg=BG_CARD).pack(anchor="e")
 
        # Nút điều khiển
        btn_frame = tk.Frame(right, bg=BG_DARK)
        btn_frame.pack(fill="x")
 
        self.start_btn = self._make_btn(
            btn_frame, "▶  BẮT ĐẦU", ACCENT, "#0f1117", self._start_camera
        )
        self.start_btn.pack(fill="x", pady=(0, 8))
 
        self.stop_btn = self._make_btn(
            btn_frame, "⏹  DỪNG", "#37474f", TEXT_PRI, self._stop_camera,
            state="disabled"
        )
        self.stop_btn.pack(fill="x", pady=(0, 8))
 
        self._make_btn(
            btn_frame, "✕  THOÁT", "#b71c1c", "#fff", self._on_close
        ).pack(fill="x")
 
        # ── Status bar ─────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Sẵn sàng  —  Nhấn BẮT ĐẦU để mở camera")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            fg=TEXT_SEC,
            bg=BG_DARK,
            anchor="w",
        ).pack(fill="x", padx=20, pady=(8, 12))
 
    # ── Widget helpers ─────────────────────────────────────────────────────────
 
    def _make_card(self, parent) -> tk.Frame:
        return tk.Frame(parent, bg=BG_CARD, padx=12, pady=10, bd=0)
 
    def _stat_card(self, parent, title: str, var: tk.StringVar, color: str) -> tk.Frame:
        card = self._make_card(parent)
        tk.Label(card, text=title, font=("Helvetica", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
        tk.Label(card, textvariable=var, font=("Courier", 30, "bold"),
                 fg=color, bg=BG_CARD).pack(anchor="w")
        return card
 
    def _make_btn(self, parent, text, bg, fg, cmd, state="normal") -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
            font=("Helvetica", 10, "bold"),
            relief="flat", bd=0, cursor="hand2",
            padx=8, pady=10, state=state,
        )
 
    def _show_placeholder(self):
        """Hiển thị màn hình chờ khi chưa có video."""
        w, h = 760, 480
        img = np.full((h, w, 3), 15, dtype=np.uint8)
        cv2.putText(img, "CAMERA CHUA MO", (w//2 - 160, h//2 - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 60, 80), 2, cv2.LINE_AA)
        cv2.putText(img, "Nhan BAT DAU de khoi dong", (w//2 - 170, h//2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 70, 90), 1, cv2.LINE_AA)
        self._display_array(img)
 
    # ── Camera controls ────────────────────────────────────────────────────────
 
    def _start_camera(self):
        if self.camera_thread and self.camera_thread.is_running:
            return
        try:
            self.camera_thread = self.camera_factory()
            self.camera_thread.on_frame = self._on_new_frame
            self.camera_thread.start()
        except Exception as exc:
            messagebox.showerror("Lỗi", f"Không thể khởi động camera:\n{exc}")
            return
 
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("🟢  Đang chạy  —  Camera đã mở")
 
    def _stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
 
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("⏹  Đã dừng  —  Nhấn BẮT ĐẦU để tiếp tục")
        self._show_placeholder()
        self._reset_stats()
 
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()
 
    # ── Frame callback (từ thread) ─────────────────────────────────────────────
 
    def _on_new_frame(self, frame: np.ndarray, stats: dict):
        """Được gọi từ CameraThread — schedule update lên main thread."""
        self._stats = stats
        # Tkinter không thread-safe → dùng after()
        self.root.after(0, self._update_ui, frame, stats)
 
    def _update_ui(self, frame: np.ndarray, stats: dict):
        self._display_array(frame)
 
        self.fps_var.set(f"{stats['fps']:.0f}")
        self.total_var.set(str(stats["total"]))
        self.mask_var.set(str(stats["mask"]))
        self.nomask_var.set(str(stats["no_mask"]))
 
        total = stats["total"]
        ratio = (stats["mask"] / total * 100) if total > 0 else 0.0
        self.progress["value"] = ratio
        self.ratio_var.set(f"{ratio:.0f} %")
 
        # Cập nhật status bar
        if total == 0:
            msg = "Không phát hiện khuôn mặt nào"
        elif stats["no_mask"] == 0:
            msg = f"✅  Tất cả {total} người đều đeo khẩu trang"
        else:
            msg = (f"⚠️  {stats['no_mask']} / {total} người KHÔNG đeo khẩu trang  "
                   f"— {stats['mask']} người đeo")
        self.status_var.set(msg)
 
    def _display_array(self, frame: np.ndarray):
        """Chuyển numpy BGR array → ImageTk và hiển thị."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize về 760×480 giữ tỷ lệ
        h, w = rgb.shape[:2]
        target_w, target_h = 760, 480
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
 
        # Padding để đủ 760×480
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - nh) // 2
        x_off = (target_w - nw) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = rgb
 
        img = Image.fromarray(canvas)
        imgtk = ImageTk.PhotoImage(image=img)
        self._latest_imgtk = imgtk  # giữ reference
        self.video_label.config(image=imgtk, width=target_w, height=target_h)
 
    def _reset_stats(self):
        for var in (self.total_var, self.mask_var, self.nomask_var):
            var.set("0")
        self.fps_var.set("—")
        self.progress["value"] = 0
        self.ratio_var.set("0 %")