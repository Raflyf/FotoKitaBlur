# Foto Kita Blur

Project iseng untuk hiburan: deteksi gestur tangan via webcam yang memicu
efek lucu di layar — gesture **peace (✌️)** memblur wajah, dan gestur lain
memunculkan emoji dan mahkota.

Semua deteksi berjalan di browser via MediaPipe (frame tidak dikirim ke
mana-mana). Ada juga CLI Python/OpenCV sebagai fallback mandiri.

## Gestur

| Gestur | Cara melakukannya | Efek |
| --- | --- | --- |
| ✌️ Peace | Telunjuk + jari tengah terangkat, manis + kelingking ditekuk | Wajah terblur |
| 🫰 Love satu tangan | Ibu jari + telunjuk direkatkan membentuk hati | Emoji love melayang |
| ❤️ Love dua tangan | Kedua tangan membentuk satu hati besar | Dua emoji love |
| 🖕 Jari tengah | Jari tengah terangkat, sisanya ditekuk | Mahkota + emoji |

Panel **Kamus Gestur** (Bahasa Indonesia) bisa dibuka dari tombol di layar
untuk melihat penjelasan tiap gestur.

## Menjalankan

### 1. Versi Desktop GUI Popup (Native Python - Rekomendasi)
```bash
python gui_app.py
```
Aplikasi GUI desktop native lengkap dengan video webcam, efek partikel emoji, audio musik dan Scuba Cat, slider blur dan sensitivity, serta panel Kamus Gestur.

### 2. Versi Web Browser
```bash
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000, izinkan akses kamera, klik **Nyalakan Kamera**.

### 3. Versi CLI Minimalis
```bash
python blur.py
```

`python blur.py --help` untuk opsi flag (kamera, confidence, kernel blur, dll).
Tekan `ESC` untuk keluar.

Musik sudah disertakan di `media/music/` (`foto-kita-blur.mp3`).

## Struktur

```
app.py                 Server Flask (serve aset + audio)
blur.py                CLI deteksi OpenCV/MediaPipe
static/main.js         Logika deteksi frontend + efek + UI
static/models/         Model MediaPipe (tangan + wajah)
templates/index.html   UI satu halaman
tests/test_blur.py     Unit test logika gestur
media/music/           Audio lokal (di-gitignore)
```

## Pengujian

```bash
python -m unittest tests.test_blur -v
node --test tests/gestures.test.mjs
```

## Teknologi

MediaPipe Tasks Vision, vanilla JavaScript, Flask, OpenCV, NumPy (versi
terkunci di `requirements.txt`).
