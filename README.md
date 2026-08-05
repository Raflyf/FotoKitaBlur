# Foto Kita Blur — Perisai Privasi dengan Gestur Peace

Kamera webmu selalu menonton. Di tengah panggilan video, live streaming, atau
kelas, hal yang paling tidak kamu inginkan adalah ketahuan lengah — momen
pribadi, layar sensitif, wajah yang tidak ingin kamu perlihatkan. Kebanyakan
alat privasi memaksamu berjibaku mencari tombol. Foto Kita Blur mengambil
pendekatan berbeda: **tanganmu adalah tombolnya.**

Tunjukkan gestur **peace** (✌️) ke kamera dan semua wajah di frame langsung
terblur, sampai tanganmu turun. Tanpa keyboard, tanpa menu, tanpa jeda. Ini
perisai privasi yang bisa kamu operasikan dengan satu gerakan, dalam waktu
setengah detik, tanpa memutus kontak mata dengan lawan bicaramu.

## Kenapa project ini ada

Ide ini lahir dari pengamatan sederhana: orang-orang yang memblur wajah saat
di kamera — streamer, mahasiswa, pekerja yang peduli privasi — semuanya
menghadapi masalah yang sama. Mengaktifkan blur adalah pekerjaan yang
bertele-tele (pindah tab, cari tombol, klik, kembali), jadi mereka akhirnya
membiarkan blur menyala permanen atau malah lupa total. Switch berbasis
gestur menghilangkan biaya penggunaan perlindungan privasi, sehingga fitur
ini benar-benar dipakai.

Project ini juga sengaja dibuat ceria: gestur peace memblur wajah, tapi gestur
lain memicu efek yang lebih ringan — hati, mahkota, bahkan kucing selam.
Privasi tidak harus terasa seperti tombol panik.

## Cara kerjanya

Semuanya berjalan di perangkatmu. Aplikasi membuka webcam dan mengumpankan
setiap frame melalui dua model MediaPipe yang dimuat di browser:

1. **Hand landmarker** — melacak 21 titik landmark per tangan, sehingga
   aplikasi bisa membedakan *jari mana yang terangkat*, bukan sekadar
   "ada tangan".
2. **Face detector** — menemukan setiap wajah di frame sehingga efek (blur,
   mahkota) bisa diterapkan tepat ke orang yang benar.

Loop deteksi berjalan sekitar 15 kali per detik. Setiap gestur memiliki logika
sendiri dengan **histeresis temporal** — gestur harus terdeteksi konsisten
selama beberapa frame berurutan sebelum memicu efek, dan tetap aktif untuk
sejenak setelah tanganmu turun, sehingga satu frame yang terlewat tidak akan
membuat efek berkedip. Saat gestur peace aktif, filter blur CSS diterapkan di
atas canvas; frame mentahnya sendiri tidak pernah disimpan atau dikirim ke
mana pun.

Jika kamu tidak ingin membuka browser sama sekali, `blur.py` adalah CLI
Python/OpenCV mandiri yang mengimplementasikan logika gestur yang persis sama
(heuristiknya dicerminkan di kedua sisi, dan dikunci oleh unit test).

## Gestur

| Gestur | Cara melakukannya | Efek |
| --- | --- | --- |
| ✌️ Peace (V) | Telunjuk dan jari tengah terangkat, manis dan kelingking ditekuk | Wajah terblur |
| 🫰 Love satu tangan | Ujung ibu jari dan telunjuk direkatkan membentuk hati kecil | Emoji love melayang |
| ❤️ Love dua tangan | Kedua tangan membentuk satu hati besar (ujung telunjuk saling menyentuh) | Dua emoji love, satu per tangan |
| 🖕 Jari tengah | Jari tengah terangkat, sisanya ditekuk | Mahkota iseng + emoji |
| 🤿 Scubacat | Satu tangan menutup hidung, tangan lain melambai cepat | Kucing selam + suara kicau |

Panel **Kamus Gestur** di dalam aplikasi (kamus Bahasa Indonesia yang bisa
ditampilkan/dimatikan, dan mengingat pilihanmu antar kunjungan) menjelaskan
setiap gestur di layar, sehingga pengguna baru tidak perlu membaca dokumen.

Deteksi sengaja dibuat ketat: gestur peace mensyaratkan telunjuk dan jari
tengah terentang penuh, manis dan kelingking terlipat, ada celah V yang
terlihat, serta ibu jari menempel — sehingga tangan yang rileks atau lambaian
tangan tidak akan pernah memblur layarmu secara tidak sengaja.

## Menjalankan web app

```bash
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000, izinkan akses kamera, klik **Start Camera**, lalu
coba gestur-gesturnya. Tanpa proses build; frontend-nya vanilla JavaScript.

Variabel lingkungan:

| Variabel | Default | Deskripsi |
| --- | --- | --- |
| `FOTO_BLUR_HOST` | `127.0.0.1` | Host yang di-bind |
| `FOTO_BLUR_PORT` | `5000` | Port yang di-bind |
| `FOTO_BLUR_DEBUG` | `0` | Set `1` untuk mode debug |

## Menjalankan CLI

```bash
python blur.py
```

| Flag | Default | Deskripsi |
| --- | --- | --- |
| `--camera` | `0` | Indeks perangkat kamera |
| `--min-confidence` | `0.5` | Ambang keyakinan deteksi |
| `--kernel` | `61` | Ukuran kernel blur Gaussian (ganjil, min 3) |
| `--no-enhance` | off | Nonaktifkan pencerahan CLAHE untuk cahaya redup |

Tekan `ESC` untuk keluar. Membutuhkan Python 3.10+ dan kamera yang berfungsi.

> Musik: web app dapat memutar musik latar (0:23–0:52) dan klip kicau untuk
> kucing selam, tetapi file audio berlisensi hak cipta dan sengaja tidak
> dilacak di git. Letakkan `foto-kita-blur.mp3` dan `kicau-mania.mp3` di
> `media/music/` untuk mengaktifkannya.

## Arsitektur

| Lapisan | Teknologi |
| --- | --- |
| Web frontend | Vanilla JS + MediaPipe Tasks Vision (`hand_landmarker.task`) |
| Backend | Flask — menyajikan aset + audio, menambah security headers |
| CLI | Python + OpenCV + MediaPipe (`blur.py`) |
| Logika gestur | Implementasi cermin di `main.js` dan `blur.py` |

Flask ada hanya untuk menyajikan aplikasi statis dan audio melalui HTTP;
semua computer vision berjalan di sisi klien, dan CLI adalah alternatif
mandiri bagi yang lebih suka terminal.

## Struktur project

```
app.py                 Server Flask + security headers + rute audio
blur.py                Detektor CLI OpenCV/MediaPipe (paritas gestur)
static/main.js         Deteksi frontend, efek, musik, UI
static/models/         File model MediaPipe tangan + wajah
templates/index.html   UI satu halaman
tests/test_blur.py     Unit test logika gestur
media/music/           Audio lokal (di-gitignore)
```

## Pengujian

```bash
python -m unittest tests.test_blur -v
```

Tes mencakup normalisasi kernel, penerimaan dan penolakan gestur peace
(termasuk tangan yang seharusnya *tidak* memicunya), heuristik jari tengah
dan finger heart, serta menjaga paritas antara logika Python dan JavaScript.

## Privasi & keamanan

- Semua frame diproses lokal di browser; tidak ada yang diunggah
- Content-Security-Policy ketat: script hanya dari self + jsdelivr, dengan
  `'wasm-unsafe-eval'` yang dibutuhkan runtime WASM MediaPipe
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`
- Rute audio menggunakan daftar putih kunci yang tetap; jalur lain
  mengembalikan 404

## Teknologi

MediaPipe Tasks Vision, vanilla JavaScript, Flask, OpenCV, NumPy — versi
terkunci di `requirements.txt` (Flask 3.1.3, mediapipe 0.10.35, numpy 2.4.6,
opencv-python 4.13.0.92).
