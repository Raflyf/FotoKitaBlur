"""Foto Kita Blur - Desktop AI Vision & Gesture GUI Application

Full Python native desktop application with integrated camera feed,
OpenCV/MediaPipe gesture recognition, animated emoji particles,
3D rotating halo crowns, Scuba Cat PiP overlay, and native audio playback.
"""

import ctypes
import math
import os
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageSequence

# Import mathematically robust gesture predicates from blur.py
from blur import (
    get_distance,
    is_finger_extended,
    is_finger_folded,
    PeaceBlurDetector
)

# Enable High-DPI scaling on Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class AudioController:
    """Manages audio playback via Windows Multimedia MCI API without external dependencies."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.music_path = self.base_dir / "media" / "music" / "foto-kita-blur.mp3"
        self.kicau_path = self.base_dir / "media" / "music" / "kicau-mania.mp3"

        self.is_music_open = False
        self.is_kicau_open = False
        self.is_music_playing = False
        self.is_kicau_playing = False

        self._lock = threading.Lock()
        self._running = True
        self._monitor_thread = threading.Thread(target=self._audio_loop_monitor, daemon=True)
        self._monitor_thread.start()

    def _mci(self, cmd):
        try:
            return ctypes.windll.winmm.mciSendStringW(cmd, None, 0, 0)
        except Exception as e:
            print("MCI Error:", e)
            return -1

    def _get_position_ms(self, alias):
        buf = ctypes.create_unicode_buffer(128)
        ret = ctypes.windll.winmm.mciSendStringW(f"status {alias} position", buf, 128, 0)
        if ret == 0:
            try:
                return int(buf.value)
            except ValueError:
                return 0
        return 0

    def play_music(self):
        with self._lock:
            if not self.music_path.exists():
                return False
            if not self.is_music_open:
                p = str(self.music_path.resolve())
                self._mci(f'open "{p}" type mpegvideo alias bgm')
                self._mci('set bgm time format milliseconds')
                self.is_music_open = True

            # Start at 23 seconds (23000 ms)
            self._mci('play bgm from 23000')
            self.is_music_playing = True
            return True

    def stop_music(self):
        with self._lock:
            if self.is_music_open:
                self._mci('stop bgm')
            self.is_music_playing = False

    def play_kicau(self):
        with self._lock:
            if not self.kicau_path.exists() or self.is_kicau_playing:
                return
            if not self.is_kicau_open:
                p = str(self.kicau_path.resolve())
                self._mci(f'open "{p}" type mpegvideo alias kicau')
                self._mci('set kicau time format milliseconds')
                self.is_kicau_open = True

            # Seek to 2:13 (133000 ms)
            self._mci('play kicau from 133000')
            self.is_kicau_playing = True

    def stop_kicau(self):
        with self._lock:
            if self.is_kicau_open and self.is_kicau_playing:
                self._mci('pause kicau')
                self.is_kicau_playing = False

    def _audio_loop_monitor(self):
        while self._running:
            time.sleep(0.1)
            # Loop bgm between 23s and 52s (23000 - 52000 ms)
            if self.is_music_playing and self.is_music_open:
                pos = self._get_position_ms("bgm")
                if pos >= 52000 or pos < 22000:
                    with self._lock:
                        self._mci('play bgm from 23000')

            # Loop kicau between 133s and 146s (133000 - 146000 ms)
            if self.is_kicau_playing and self.is_kicau_open:
                pos = self._get_position_ms("kicau")
                if pos >= 146000 or pos < 132000:
                    with self._lock:
                        self._mci('play kicau from 133000')

    def cleanup(self):
        self._running = False
        with self._lock:
            if self.is_music_open:
                self._mci('close bgm')
                self.is_music_open = False
            if self.is_kicau_open:
                self._mci('close kicau')
                self.is_kicau_open = False


class Particle:
    """Floating animated emoji particle."""

    def __init__(self, x, y, emojis=None):
        self.x = float(x)
        self.y = float(y)
        self.emoji = random.choice(emojis) if emojis else random.choice(['💖', '❤️', '💕', '💗', '💓', '💝'])
        self.size = random.randint(22, 38)
        self.alpha = 1.0
        self.fade_rate = random.uniform(0.022, 0.038)
        self.vx = random.uniform(-2.0, 2.0)
        self.vy = random.uniform(-4.5, -2.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.08  # Gravity dampener
        self.alpha -= self.fade_rate

    @property
    def is_alive(self):
        return self.alpha > 0.0


class FotoKitaBlurApp:
    """Main Desktop GUI Application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Foto Kita Blur - AI Privacy Shield")
        self.root.geometry("1100x820")
        self.root.minsize(980, 720)
        self.root.configure(bg="#050507")

        self.base_dir = Path(__file__).resolve().parent
        self.audio = AudioController(self.base_dir)

        # Vision Models
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=0.45,
            model_selection=0
        )

        # Camera & Processing State
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None

        # Gesture & Temporal Filters
        self.peace_hold_frames = 0
        self.scubacat_hold_frames = 0
        self.waving_energy = 0.0
        self.last_wave_wrist_x = None
        self.last_wave_wrist_y = None
        self.wave_stroke_dir = 0
        self.wave_stroke_dist = 0.0

        # Face tracking smoothing (EMA)
        self.tracked_face = None  # (cx, cy, w, h)
        self.face_missed_frames = 0

        # Particles & Crown
        self.particles = []
        self.crown_angle = 0.0
        self.last_crown_time = time.perf_counter()

        # Load Scuba Cat GIF frames
        self.cat_frames = []
        self.cat_frame_idx = 0
        self.cat_last_time = time.perf_counter()
        self._load_cat_gif()

        # Fonts
        self.emoji_font = None
        self.text_font = None
        self._load_fonts()

        # FPS metrics
        self.fps = 0.0
        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0

        # UI State Variables
        self.var_blur = tk.IntVar(value=25)
        self.var_conf = tk.IntVar(value=50)
        self.var_skeleton = tk.BooleanVar(value=False)
        self.var_crown = tk.BooleanVar(value=False)
        self.var_cheeky = tk.BooleanVar(value=False)

        self._build_ui()

        # Window Close Protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_fonts(self):
        try:
            self.emoji_font = ImageFont.truetype("seguiemj.ttf", 36)
            self.emoji_font_small = ImageFont.truetype("seguiemj.ttf", 26)
            self.text_font = ImageFont.truetype("arial.ttf", 16)
            self.title_font = ImageFont.truetype("arialbd.ttf", 18)
        except Exception as e:
            print("Font loading fallback:", e)
            self.emoji_font = ImageFont.load_default()
            self.emoji_font_small = ImageFont.load_default()
            self.text_font = ImageFont.load_default()
            self.title_font = ImageFont.load_default()

    def _load_cat_gif(self):
        gif_path = self.base_dir / "static" / "cat-scuba.gif"
        if gif_path.exists():
            try:
                img = Image.open(gif_path)
                for frame in ImageSequence.Iterator(img):
                    f = frame.copy().convert("RGBA")
                    f = f.resize((160, 160), Image.Resampling.BILINEAR)
                    self.cat_frames.append(f)
            except Exception as e:
                print("Failed loading cat gif:", e)

    def _build_ui(self):
        # 1. Top Header
        header = tk.Frame(self.root, bg="#0e0e12", height=65, padx=20, pady=10)
        header.pack(fill="x", side="top")

        title_frame = tk.Frame(header, bg="#0e0e12")
        title_frame.pack(side="left")

        lbl_sub = tk.Label(
            title_frame, text="AI PRIVACY SHIELD & GESTURE SYSTEM",
            fg="#90909c", bg="#0e0e12", font=("Segoe UI", 9, "bold")
        )
        lbl_sub.pack(anchor="w")

        lbl_title = tk.Label(
            title_frame, text="Foto Kita Blur (Desktop GUI)",
            fg="#ffffff", bg="#0e0e12", font=("Segoe UI", 14, "bold")
        )
        lbl_title.pack(anchor="w")

        self.lbl_status = tk.Label(
            header, text="Kamera Nonaktif", fg="#eab308", bg="#1a1a24",
            font=("Segoe UI", 10, "bold"), padx=14, pady=5
        )
        self.lbl_status.pack(side="right")

        # 2. Main Center Area (Video Frame + Controls)
        content_frame = tk.Frame(self.root, bg="#050507", padx=16, pady=12)
        content_frame.pack(fill="both", expand=True)

        # Video Canvas Frame
        video_wrapper = tk.Frame(content_frame, bg="#000000", bd=1, relief="solid")
        video_wrapper.pack(fill="both", expand=True)

        self.canvas = tk.Label(video_wrapper, bg="#000000")
        self.canvas.pack(fill="both", expand=True)

        # 3. Action Buttons Row
        action_bar = tk.Frame(content_frame, bg="#050507", pady=10)
        action_bar.pack(fill="x")

        self.btn_camera = tk.Button(
            action_bar, text="Nyalakan Kamera", bg="#ffffff", fg="#050507",
            activebackground="#e4e4e7", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2", command=self.toggle_camera
        )
        self.btn_camera.pack(side="left", padx=5, expand=True, fill="x")

        self.btn_music = tk.Button(
            action_bar, text="Musik (0:23 - 0:52)", bg="#16161c", fg="#f4f4f7",
            activebackground="#252530", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=8, cursor="hand2", command=self.toggle_music
        )
        self.btn_music.pack(side="left", padx=5, expand=True, fill="x")

        self.btn_glossary = tk.Button(
            action_bar, text="Kamus Gestur", bg="#16161c", fg="#f4f4f7",
            activebackground="#252530", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=8, cursor="hand2", command=self.show_glossary
        )
        self.btn_glossary.pack(side="left", padx=5, expand=True, fill="x")

        # 4. Settings Panel
        settings_frame = tk.Frame(content_frame, bg="#0e0e12", bd=1, relief="solid", padx=14, pady=10)
        settings_frame.pack(fill="x", pady=4)

        # Sliders Row
        sliders_row = tk.Frame(settings_frame, bg="#0e0e12")
        sliders_row.pack(fill="x", pady=4)

        # Blur Slider
        lbl_blur_title = tk.Label(sliders_row, text="Blur Intensity:", fg="#90909c", bg="#0e0e12", font=("Segoe UI", 9))
        lbl_blur_title.pack(side="left", padx=(0, 6))
        self.scale_blur = ttk.Scale(sliders_row, from_=5, to=80, orient="horizontal", variable=self.var_blur)
        self.scale_blur.pack(side="left", padx=6, fill="x", expand=True)
        self.lbl_blur_val = tk.Label(sliders_row, text="25px", fg="#ffffff", bg="#0e0e12", font=("Segoe UI", 9, "bold"), width=5)
        self.lbl_blur_val.pack(side="left")
        self.scale_blur.configure(command=lambda v: self.lbl_blur_val.config(text=f"{int(float(v))}px"))

        # Confidence Slider
        lbl_conf_title = tk.Label(sliders_row, text="Sensitivity:", fg="#90909c", bg="#0e0e12", font=("Segoe UI", 9))
        lbl_conf_title.pack(side="left", padx=(20, 6))
        self.scale_conf = ttk.Scale(sliders_row, from_=20, to=90, orient="horizontal", variable=self.var_conf)
        self.scale_conf.pack(side="left", padx=6, fill="x", expand=True)
        self.lbl_conf_val = tk.Label(sliders_row, text="50%", fg="#ffffff", bg="#0e0e12", font=("Segoe UI", 9, "bold"), width=5)
        self.lbl_conf_val.pack(side="left")
        self.scale_conf.configure(command=lambda v: self.lbl_conf_val.config(text=f"{int(float(v))}%"))

        # Toggles Row
        toggles_row = tk.Frame(settings_frame, bg="#0e0e12")
        toggles_row.pack(fill="x", pady=6)

        chk_skel = tk.Checkbutton(
            toggles_row, text="Show Hand Skeleton", variable=self.var_skeleton,
            fg="#f4f4f7", bg="#0e0e12", selectcolor="#16161c", activebackground="#0e0e12",
            font=("Segoe UI", 9, "bold")
        )
        chk_skel.pack(side="left", padx=10)

        chk_crown = tk.Checkbutton(
            toggles_row, text="Mahkota Love Halo", variable=self.var_crown,
            command=self._on_crown_toggle,
            fg="#f4f4f7", bg="#0e0e12", selectcolor="#16161c", activebackground="#0e0e12",
            font=("Segoe UI", 9, "bold")
        )
        chk_crown.pack(side="left", padx=10)

        chk_cheeky = tk.Checkbutton(
            toggles_row, text="Mahkota Jari Tengah Halo", variable=self.var_cheeky,
            command=self._on_cheeky_toggle,
            fg="#f4f4f7", bg="#0e0e12", selectcolor="#16161c", activebackground="#0e0e12",
            font=("Segoe UI", 9, "bold")
        )
        chk_cheeky.pack(side="left", padx=10)

        # 5. Diagnostics Panel
        diag_frame = tk.Frame(content_frame, bg="#09090d", bd=1, relief="solid", padx=10, pady=6)
        diag_frame.pack(fill="x", pady=2)

        lbl_diag = tk.Label(diag_frame, text="DIAGNOSTIK JARI (TANGAN UTAMA):", fg="#5f5f6e", bg="#09090d", font=("Segoe UI", 8, "bold"))
        lbl_diag.pack(side="left", padx=(0, 10))

        self.diag_badges = {}
        for finger in ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]:
            f_frame = tk.Frame(diag_frame, bg="#16161c", padx=8, pady=2)
            f_frame.pack(side="left", padx=4)
            lbl_name = tk.Label(f_frame, text=finger, fg="#90909c", bg="#16161c", font=("Segoe UI", 8))
            lbl_name.pack(side="left", padx=(0, 4))
            lbl_val = tk.Label(f_frame, text="FOLDED", fg="#5f5f6e", bg="#16161c", font=("Segoe UI", 8, "bold"))
            lbl_val.pack(side="left")
            self.diag_badges[finger] = lbl_val

    def _on_crown_toggle(self):
        if self.var_crown.get():
            self.var_cheeky.set(False)

    def _on_cheeky_toggle(self):
        if self.var_cheeky.get():
            self.var_crown.set(False)

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error Kamera", "Tidak dapat membuka kamera web index 0.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_camera_running = True
        self.btn_camera.config(text="Matikan Kamera", bg="#f43f5e", fg="#ffffff")
        self.lbl_status.config(text="Kamera Aktif", fg="#10b981")

        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0

        # Start Video Update Loop in Tkinter
        self.root.after(10, self._process_frame)

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.audio.stop_kicau()
        self.scubacat_hold_frames = 0
        self.peace_hold_frames = 0
        self.particles.clear()

        # Reset Canvas
        blank = np.zeros((480, 854, 3), dtype=np.uint8)
        img = ImageTk.PhotoImage(image=Image.fromarray(blank))
        self.canvas.configure(image=img)
        self.canvas.image = img

        self.btn_camera.config(text="Nyalakan Kamera", bg="#ffffff", fg="#050507")
        self.lbl_status.config(text="Kamera Nonaktif", fg="#eab308")

        for lbl in self.diag_badges.values():
            lbl.config(text="FOLDED", fg="#5f5f6e")

    def toggle_music(self):
        if self.audio.is_music_playing:
            self.audio.stop_music()
            self.btn_music.config(text="Musik (0:23 - 0:52)", bg="#16161c", fg="#f4f4f7")
        else:
            ok = self.audio.play_music()
            if ok:
                self.btn_music.config(text="Stop Musik", bg="#f43f5e", fg="#ffffff")
            else:
                messagebox.showwarning("File Tidak Ditemukan", "File audio media/music/foto-kita-blur.mp3 tidak ditemukan.")

    def show_glossary(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Kamus Gestur Interaktif")
        dlg.geometry("560x520")
        dlg.configure(bg="#0e0e12")
        dlg.transient(self.root)

        tk.Label(
            dlg, text="Kamus Gestur Foto Kita Blur",
            fg="#ffffff", bg="#0e0e12", font=("Segoe UI", 13, "bold"), pady=12
        ).pack()

        cards = [
            ("✌️", "Peace (Dua Jari)", "Angkat telunjuk & tengah membentuk V, tekuk jari lainnya.", "Efek: Wajah & Layar Terblur"),
            ("🫰", "Love Satu Tangan", "Dekatkan/silangkan ujung ibu jari & telunjuk ala Korea.", "Efek: Partikel Love Melayang"),
            ("🫶", "Love Dua Tangan", "Bentuk hati: satukan ujung telunjuk di atas dan ibu jari di bawah.", "Efek: Partikel Hati Besar"),
            ("🖕", "Jari Tengah", "Angkat jari tengah tegak, tekuk jari lainnya.", "Efek: Partikel & Mahkota Cheeky"),
            ("🤿🐱", "Tutup Hidung & Melambai", "Tutup hidung dengan satu tangan, lalu lambaikan tangan lainnya.", "Efek: Kucing Scuba + Lagu Kicau Mania")
        ]

        for emoji, name, desc, eff in cards:
            c = tk.Frame(dlg, bg="#16161c", bd=1, relief="solid", padx=12, pady=8)
            c.pack(fill="x", padx=16, pady=5)
            tk.Label(c, text=emoji, font=("Segoe UI", 20), fg="#ffffff", bg="#16161c").pack(side="left", padx=(0, 12))
            info = tk.Frame(c, bg="#16161c")
            info.pack(side="left", fill="both")
            tk.Label(info, text=name, font=("Segoe UI", 10, "bold"), fg="#ffffff", bg="#16161c").pack(anchor="w")
            tk.Label(info, text=desc, font=("Segoe UI", 8), fg="#90909c", bg="#16161c").pack(anchor="w")
            tk.Label(info, text=eff, font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg="#16161c").pack(anchor="w")

        tk.Button(
            dlg, text="Tutup", bg="#ffffff", fg="#050507", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2", command=dlg.destroy
        ).pack(pady=12)

    def _process_frame(self):
        if not self.is_camera_running or self.cap is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.root.after(15, self._process_frame)
            return

        # Flip frame horizontally for natural mirror look
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update FPS
        self.frame_counter += 1
        now = time.perf_counter()
        if now - self.last_fps_time >= 1.0:
            self.fps = (self.frame_counter / (now - self.last_fps_time))
            self.frame_counter = 0
            self.last_fps_time = now

        # Run MediaPipe Hands & Face Detection
        hands_result = self.hands_detector.process(rgb)
        face_result = self.face_detector.process(rgb)

        # 1. Track Face with Smoothing (EMA)
        raw_faces = []
        if face_result.detections:
            for det in face_result.detections:
                box = det.location_data.relative_bounding_box
                raw_faces.append((
                    box.xmin + box.width / 2.0,
                    box.ymin + box.height / 2.0,
                    box.width,
                    box.height
                ))

        if raw_faces:
            rf = raw_faces[0]
            if self.tracked_face is None:
                self.tracked_face = list(rf)
            else:
                alpha = 0.35
                self.tracked_face[0] += (rf[0] - self.tracked_face[0]) * alpha
                self.tracked_face[1] += (rf[1] - self.tracked_face[1]) * alpha
                self.tracked_face[2] += (rf[2] - self.tracked_face[2]) * alpha
                self.tracked_face[3] += (rf[3] - self.tracked_face[3]) * alpha
            self.face_missed_frames = 0
        else:
            self.face_missed_frames += 1
            if self.face_missed_frames > 6:
                self.tracked_face = None

        # 2. Gesture Evaluation
        peace_detected = False
        finger_heart_spawns = []
        cheeky_spawns = []
        two_hand_heart_spawn = None
        hands_landmarks = hands_result.multi_hand_landmarks or []

        if hands_landmarks:
            # Diagnostics on First Hand
            h0 = hands_landmarks[0].landmark
            palm_size = get_distance(h0[0], h0[9])

            self.diag_badges["Jempol"].config(
                text="UP" if get_distance(h0[4], h0[5]) > palm_size * 0.60 else "FOLDED",
                fg="#10b981" if get_distance(h0[4], h0[5]) > palm_size * 0.60 else "#5f5f6e"
            )
            self.diag_badges["Telunjuk"].config(
                text="UP" if is_finger_extended(h0, 5, 6, 8) else "FOLDED",
                fg="#10b981" if is_finger_extended(h0, 5, 6, 8) else "#5f5f6e"
            )
            self.diag_badges["Tengah"].config(
                text="UP" if is_finger_extended(h0, 9, 10, 12) else "FOLDED",
                fg="#10b981" if is_finger_extended(h0, 9, 10, 12) else "#5f5f6e"
            )
            self.diag_badges["Manis"].config(
                text="UP" if is_finger_extended(h0, 13, 14, 16) else "FOLDED",
                fg="#10b981" if is_finger_extended(h0, 13, 14, 16) else "#5f5f6e"
            )
            self.diag_badges["Kelingking"].config(
                text="UP" if is_finger_extended(h0, 17, 18, 20) else "FOLDED",
                fg="#10b981" if is_finger_extended(h0, 17, 18, 20) else "#5f5f6e"
            )

            # Check individual hands: Middle finger takes strict precedence over peace
            middle_finger_detected = False
            for hand in hands_landmarks:
                lms = hand.landmark
                if PeaceBlurDetector.is_middle_finger(lms):
                    middle_finger_detected = True
                    gx = lms[12].x * w
                    gy = lms[12].y * h
                    cheeky_spawns.append((gx, gy))

                if PeaceBlurDetector.is_finger_heart(lms):
                    gx = (lms[8].x + lms[4].x) / 2.0 * w
                    gy = (lms[8].y + lms[4].y) / 2.0 * h
                    finger_heart_spawns.append((gx, gy))

            # Only check peace if NO hand is showing middle finger
            if not middle_finger_detected:
                for hand in hands_landmarks:
                    if PeaceBlurDetector.is_peace(hand.landmark):
                        peace_detected = True
                        break

            # Two-Hand Heart
            if len(hands_landmarks) >= 2:
                two_res = PeaceBlurDetector.is_two_hand_heart(
                    hands_landmarks[0].landmark, hands_landmarks[1].landmark
                )
                if two_res:
                    two_hand_heart_spawn = (two_res["x"] * w, two_res["y"] * h)

            # Check Scuba Cat (Nose Hold + Waving)
            if self.tracked_face is not None and len(hands_landmarks) >= 2:
                fcx, fcy, fw, fh = self.tracked_face
                nose_hand_idx = -1

                # 1. Identify which hand is near the face (nose/mouth area)
                for i, hand in enumerate(hands_landmarks):
                    hl = hand.landmark
                    # Check key hand landmarks: wrist(0), palm(9), index tip(8), thumb tip(4)
                    d0 = math.hypot(hl[0].x - fcx, hl[0].y - fcy)
                    d9 = math.hypot(hl[9].x - fcx, hl[9].y - fcy)
                    d8 = math.hypot(hl[8].x - fcx, hl[8].y - fcy)
                    d4 = math.hypot(hl[4].x - fcx, hl[4].y - fcy)
                    min_dist = min(d0, d9, d8, d4)
                    if min_dist < fw * 0.85:
                        nose_hand_idx = i
                        break

                # 2. If hand is at face, update waving integrator on the other hand
                if nose_hand_idx != -1:
                    wave_hand = hands_landmarks[0 if nose_hand_idx == 1 else 1].landmark
                    nose_wrist = hands_landmarks[nose_hand_idx].landmark[0]
                    wave_wrist = wave_hand[0]

                    # Separation checks: waving hand is away from face and away from nose hand
                    dist_wave_face = math.hypot(wave_wrist.x - fcx, wave_wrist.y - fcy)
                    dist_between_hands = math.hypot(wave_wrist.x - nose_wrist.x, wave_wrist.y - nose_wrist.y)

                    if dist_wave_face > fw * 0.45 and dist_between_hands > fw * 0.45:
                        if self.last_wave_wrist_x is None:
                            self.last_wave_wrist_x = wave_wrist.x
                            self.last_wave_wrist_y = wave_wrist.y

                        dx = wave_wrist.x - self.last_wave_wrist_x
                        dy = wave_wrist.y - self.last_wave_wrist_y
                        self.last_wave_wrist_x = wave_wrist.x
                        self.last_wave_wrist_y = wave_wrist.y

                        dist = math.hypot(dx, dy)
                        if dist < 0.007:
                            self.waving_energy = max(0.0, self.waving_energy - 1.5)
                            self.wave_stroke_dist *= 0.85
                        else:
                            primary_delta = dx if abs(dx) >= abs(dy) else dy
                            d = 1 if primary_delta > 0.004 else (-1 if primary_delta < -0.004 else 0)
                            if d != 0:
                                if self.wave_stroke_dir == 0:
                                    self.wave_stroke_dir = d
                                    self.wave_stroke_dist = abs(primary_delta)
                                elif d == self.wave_stroke_dir:
                                    self.wave_stroke_dist += abs(primary_delta)
                                else:
                                    # Direction reversed!
                                    min_stroke = min(fw * 0.15, 0.028)
                                    if self.wave_stroke_dist >= min_stroke:
                                        self.waving_energy = min(100.0, self.waving_energy + 24.0)
                                    self.wave_stroke_dir = d
                                    self.wave_stroke_dist = abs(primary_delta)
                    else:
                        self.waving_energy = max(0.0, self.waving_energy - 2.0)
                        self.last_wave_wrist_x = None
                        self.wave_stroke_dist = 0.0
                        self.wave_stroke_dir = 0
                else:
                    self.waving_energy = max(0.0, self.waving_energy - 2.0)
                    self.last_wave_wrist_x = None
                    self.wave_stroke_dist = 0.0
                    self.wave_stroke_dir = 0
            else:
                self.waving_energy = max(0.0, self.waving_energy - 2.0)
                self.last_wave_wrist_x = None
                self.wave_stroke_dist = 0.0
                self.wave_stroke_dir = 0

            if self.waving_energy >= 20.0:
                self.scubacat_hold_frames = 20
        else:
            for lbl in self.diag_badges.values():
                lbl.config(text="FOLDED", fg="#5f5f6e")

        # 3. Apply Gaussian Blur on Peace Gesture
        if peace_detected:
            self.peace_hold_frames = 8
        elif self.peace_hold_frames > 0:
            self.peace_hold_frames -= 1

        is_blurring = self.peace_hold_frames > 0
        if is_blurring:
            ksize = int(self.var_blur.get())
            if ksize % 2 == 0:
                ksize += 1
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        # 4. Draw Hand Skeleton (if enabled)
        if self.var_skeleton.get() and hands_landmarks:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            for hand in hands_landmarks:
                pts = [(int(pt.x * w), int(pt.y * h)) for pt in hand.landmark]
                for p1, p2 in connections:
                    cv2.line(frame, pts[p1], pts[p2], (246, 130, 59), 3, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Convert OpenCV BGR to PIL Image for Emoji & Overlay Rendering
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 5. Spawn and Render Particles
        for sx, sy in finger_heart_spawns:
            self.particles.append(Particle(sx, sy, ['💖', '❤️', '💕', '💗', '💓', '💝']))

        if two_hand_heart_spawn:
            self.particles.append(Particle(two_hand_heart_spawn[0], two_hand_heart_spawn[1], ['💖', '❤️', '💝']))

        for sx, sy in cheeky_spawns:
            self.particles.append(Particle(sx, sy, ['🖕', '😜', '🤪', '😝', '👅']))

        # Update particles
        self.particles = [p for p in self.particles if p.is_alive]
        for p in self.particles:
            p.update()
            if p.is_alive:
                draw.text((int(p.x), int(p.y)), p.emoji, font=self.emoji_font, embedded_color=True)

        # 6. Draw 3D Halo Crown (Permanent per toggle switch)
        show_crown = self.var_crown.get()
        show_cheeky = self.var_cheeky.get()

        if (show_crown or show_cheeky) and self.tracked_face is not None:
            delta = now - self.last_crown_time
            self.crown_angle += 2.4 * delta
            self.last_crown_time = now

            emojis = ['🖕', '😜', '🤪', '🖕', '😝', '👅'] if show_cheeky else ['💖', '❤️', '💕', '💗', '💓', '💝']
            fcx, fcy, fw, fh = self.tracked_face
            rx = fw * w * 0.58
            ry = fh * h * 0.12
            center_x = fcx * w
            center_y = (fcy - fh * 0.72) * h

            # Depth perspective sorting
            halo_items = []
            for i in range(6):
                ang = self.crown_angle + (i * 2.0 * math.pi / 6.0)
                sin_a = math.sin(ang)
                cos_a = math.cos(ang)
                hx = center_x + cos_a * rx
                hy = center_y + sin_a * ry
                halo_items.append((sin_a, hx, hy, emojis[i % len(emojis)]))

            halo_items.sort(key=lambda t: t[0])  # Back items first
            for _, hx, hy, em in halo_items:
                draw.text((int(hx - 18), int(hy - 18)), em, font=self.emoji_font, embedded_color=True)

            # Ambient particles only when actively triggered
            if random.random() < 0.25:
                self.particles.append(Particle(random.uniform(0, w), random.uniform(0, h * 0.7), emojis))
        else:
            self.last_crown_time = now

        # 7. Scuba Cat Overlay & Audio
        if self.scubacat_hold_frames > 0:
            self.scubacat_hold_frames -= 1
            self.audio.play_kicau()

            if self.cat_frames:
                cat_now = time.perf_counter()
                if cat_now - self.cat_last_time > 0.035:
                    self.cat_frame_idx = (self.cat_frame_idx + 1) % len(self.cat_frames)
                    self.cat_last_time = cat_now

                cat_img = self.cat_frames[self.cat_frame_idx]
                cw, ch = cat_img.size
                paste_x = w - cw - 24
                paste_y = h - ch - 24
                # Draw sleek badge background behind cat
                draw.rounded_rectangle(
                    [paste_x - 6, paste_y - 6, paste_x + cw + 6, paste_y + ch + 6],
                    radius=16, fill=(14, 14, 18, 200), outline=(255, 255, 255, 120), width=2
                )
                pil_img.paste(cat_img, (paste_x, paste_y), cat_img)
        else:
            self.audio.stop_kicau()

        # 8. Render HUD Badges Directly on Canvas
        active_gesture_text = "MEMINDAI GESTUR..."
        badge_color = (255, 255, 255)

        if self.scubacat_hold_frames > 0:
            active_gesture_text = "🐱 KICAU SCUBA CAT ACTIVE"
            badge_color = (250, 204, 21)
        elif is_blurring:
            active_gesture_text = "✌️ PEACE (BLUR ACTIVE)"
            badge_color = (52, 211, 153)
        elif two_hand_heart_spawn:
            active_gesture_text = "🫶 TWO-HAND HEART"
            badge_color = (251, 113, 133)
        elif finger_heart_spawns:
            active_gesture_text = "🫰 FINGER HEART"
            badge_color = (251, 113, 133)
        elif cheeky_spawns:
            active_gesture_text = "🖕 JARI TENGAH"
            badge_color = (250, 204, 21)

        # Draw HUD pill
        draw.rounded_rectangle([16, 16, 260, 48], radius=8, fill=(14, 14, 18), outline=(255, 255, 255), width=1)
        draw.text((28, 24), active_gesture_text, fill=badge_color, font=self.text_font)

        # Draw FPS pill
        draw.rounded_rectangle([w - 110, 16, w - 16, 48], radius=8, fill=(14, 14, 18), outline=(255, 255, 255), width=1)
        draw.text((w - 96, 24), f"FPS: {int(self.fps)}", fill=(144, 144, 156), font=self.text_font)

        # Resize PIL image dynamically to fit Tkinter label
        cw = max(640, self.canvas.winfo_width())
        ch = max(360, self.canvas.winfo_height())
        if cw > 10 and ch > 10:
            scale = min(cw / w, ch / h)
            nw = int(w * scale)
            nh = int(h * scale)
            pil_img = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)

        # Display on Canvas
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.configure(image=tk_img)
        self.canvas.image = tk_img

        # Loop at target 30-60 FPS
        self.root.after(10, self._process_frame)

    def on_close(self):
        self.stop_camera()
        self.audio.cleanup()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FotoKitaBlurApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
