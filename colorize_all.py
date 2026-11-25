import argparse
import os
import time
import cv2
import numpy as np

# ---------- Lokasi berkas model ----------
PROTO  = "model/colorization_deploy_v2.prototxt"
WEIGHT = "model/colorization_release_v2.caffemodel"
POINTS = "model/pts_in_hull.npy"

EKSTENSI_FOTO  = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
EKSTENSI_VIDEO = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


# =====================================================================
#  INTI MODEL
# =====================================================================
def muat_jaringan():
    """Memuat model Caffe dan menyisipkan 313 pusat klaster warna."""
    net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHT)
    pusat = np.load(POINTS)                      # bentuk (313, 2)

    layer_kelas = net.getLayerId("class8_ab")
    layer_conv  = net.getLayerId("conv8_313_rh")
    pts = pusat.transpose().reshape(2, 313, 1, 1)
    net.getLayer(layer_kelas).blobs = [pts.astype("float32")]
    net.getLayer(layer_conv).blobs  = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def prediksi_ab(net, citra_bgr):
    """Mengembalikan kanal ab (56x56x2) hasil prediksi untuk satu citra/frame."""
    lab = cv2.cvtColor(citra_bgr.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L_masukan = cv2.resize(lab, (224, 224))[:, :, 0] - 50
    net.setInput(cv2.dnn.blobFromImage(L_masukan))
    return net.forward()[0].transpose((1, 2, 0))


def gabung_warna(citra_bgr, ab):
    """Gabungkan L asli dengan ab (boleh dari frame lain) -> BGR uint8."""
    tinggi, lebar = citra_bgr.shape[:2]
    lab = cv2.cvtColor(citra_bgr.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L_asli = lab[:, :, 0]

    ab = cv2.resize(ab, (lebar, tinggi))
    lab_baru = np.concatenate((L_asli[:, :, np.newaxis], ab), axis=2)
    bgr = cv2.cvtColor(lab_baru, cv2.COLOR_LAB2BGR)
    bgr = np.clip(bgr, 0, 1)
    return (bgr * 255).astype("uint8")


# =====================================================================
#  MODE FOTO
# =====================================================================
def proses_foto(masukan, keluaran, tampil=True):
    citra = cv2.imread(masukan)
    if citra is None:
        raise FileNotFoundError(f"Foto tidak ditemukan / tidak bisa dibaca: {masukan}")

    print("Memuat model...")
    net = muat_jaringan()

    print("Mewarnai foto...")
    hasil = gabung_warna(citra, prediksi_ab(net, citra))

    # imwrite menentukan format dari ekstensi; tanpa ekstensi valid akan error
    ekst = os.path.splitext(keluaran)[1].lower()
    if ekst not in EKSTENSI_FOTO:
        keluaran = keluaran + ".png" if not ekst else os.path.splitext(keluaran)[0] + ".png"
        print(f"Ekstensi output tidak valid, disimpan sebagai: {keluaran}")

    if not cv2.imwrite(keluaran, hasil):
        raise RuntimeError(f"Gagal menyimpan hasil ke: {keluaran}")
    print(f"Selesai! Hasil disimpan di: {keluaran}")

    if tampil:
        kiri  = cv2.resize(citra, (640, 640))
        kanan = cv2.resize(hasil, (640, 640))
        cv2.imshow("Grayscale -> Warna (tekan tombol apa saja untuk menutup)",
                   cv2.hconcat([kiri, kanan]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =====================================================================
#  MODE VIDEO
# =====================================================================
def proses_video(masukan, keluaran, tampil=True, alpha=0.8, setiap_n=1, paksa_fps=None):
    """
    alpha    : smoothing temporal (0 = tanpa smoothing, mendekati 1 = sangat stabil)
    setiap_n : prediksi ab hanya tiap N frame (1 = setiap frame)
    paksa_fps: paksa nilai FPS output (None = ikuti video asli)
    """
    cap = cv2.VideoCapture(masukan)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak dapat dibuka: {masukan}")

    fps_asli = cap.get(cv2.CAP_PROP_FPS)
    # Validasi FPS agar tidak terbaca ratusan/ribuan FPS akibat metadata VFR
    if paksa_fps is not None:
        fps = paksa_fps
    elif fps_asli is None or fps_asli <= 1 or fps_asli > 120:
        fps = 30.0
    else:
        fps = fps_asli

    lebar  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Info Video -> Resolusi: {lebar}x{tinggi} | FPS dipakai: {fps:.2f} "
          f"(metadata: {fps_asli:.2f}) | Total frame: {total}")

    penulis = cv2.VideoWriter(
        keluaran, cv2.VideoWriter_fourcc(*"mp4v"), fps, (lebar, tinggi)
    )
    if not penulis.isOpened():
        # Cadangan: AVI + XVID
        keluaran = os.path.splitext(keluaran)[0] + ".avi"
        print(f"mp4v gagal, mencoba XVID -> {keluaran}")
        penulis = cv2.VideoWriter(
            keluaran, cv2.VideoWriter_fourcc(*"XVID"), fps, (lebar, tinggi)
        )
        if not penulis.isOpened():
            cap.release()
            raise RuntimeError("Gagal membuat file video keluaran (mp4v maupun XVID)")

    print("Memuat model...")
    net = muat_jaringan()
    ab_halus = None
    nomor = 0
    mulai = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if nomor % setiap_n == 0 or ab_halus is None:
                ab_baru = prediksi_ab(net, frame)
                if ab_halus is None:
                    ab_halus = ab_baru
                else:
                    ab_halus = alpha * ab_halus + (1 - alpha) * ab_baru

            hasil = gabung_warna(frame, ab_halus)
            penulis.write(hasil)
            nomor += 1

            if nomor % 20 == 0:
                persen = f"{nomor / total * 100:.1f}%" if total > 0 else "-"
                fps_proses = nomor / (time.time() - mulai)
                print(f"Frame {nomor}/{total} ({persen}) | "
                      f"Kecepatan proses: {fps_proses:.1f} fps")

            if tampil:
                kiri  = cv2.resize(frame, (480, 480))
                kanan = cv2.resize(hasil, (480, 480))
                cv2.imshow("Asli (kiri) vs Warna (kanan) - tekan q untuk berhenti",
                           cv2.hconcat([kiri, kanan]))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Dihentikan oleh pengguna.")
                    break
    finally:
        # Selalu ditutup rapi supaya file tidak rusak
        cap.release()
        penulis.release()
        cv2.destroyAllWindows()

    durasi_akhir = nomor / fps
    print(f"Selesai! Total frame tertulis: {nomor}/{total} "
          f"(~{durasi_akhir:.1f} detik) -> {keluaran}")


# =====================================================================
#  MENU & CLI
# =====================================================================
def nama_keluaran(masukan, jenis):
    """Buat nama file keluaran otomatis, mis. foto1.png -> foto1_colorized.png"""
    dasar, ekst = os.path.splitext(masukan)
    if jenis == "foto":
        return f"{dasar}_colorized{ekst if ekst.lower() in EKSTENSI_FOTO else '.png'}"
    return f"{dasar}_colorized.mp4"


def deteksi_jenis(path):
    ekst = os.path.splitext(path)[1].lower()
    if ekst in EKSTENSI_FOTO:
        return "foto"
    if ekst in EKSTENSI_VIDEO:
        return "video"
    return None


def minta_path(prompt, default):
    teks = input(f"{prompt} [{default}]: ").strip().strip('"').strip("'")
    teks = teks.replace("\\ ", " ")   # drag-and-drop file di Terminal macOS
    return teks or default


def menu_interaktif():
    print("=" * 46)
    print("   PEWARNAAN HITAM-PUTIH (Colorization)")
    print("=" * 46)
    print("  1. Warnai FOTO")
    print("  2. Warnai VIDEO")
    print("  0. Keluar")
    pilihan = input("Pilih [1/2/0]: ").strip()

    if pilihan == "1":
        masukan = minta_path("Path foto", "images/img1.png")
        keluaran = minta_path("Path hasil", nama_keluaran(masukan, "foto"))
        proses_foto(masukan, keluaran, tampil=True)

    elif pilihan == "2":
        masukan = minta_path("Path video", "videos/video1.mp4")
        keluaran = minta_path("Path hasil", nama_keluaran(masukan, "video"))
        lihat = input("Tampilkan pratinjau saat proses? [y/N]: ").strip().lower() == "y"
        proses_video(masukan, keluaran, tampil=lihat)

    elif pilihan == "0":
        print("Keluar.")
    else:
        print("Pilihan tidak valid.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pewarnaan foto/video hitam-putih. "
                    "Jalankan tanpa argumen untuk memakai menu interaktif."
    )
    ap.add_argument("-i", "--input", default=None, help="path foto/video masukan")
    ap.add_argument("-o", "--output", default=None, help="path hasil keluaran")
    ap.add_argument("-m", "--mode", choices=["foto", "video"], default=None,
                    help="paksa mode (default: deteksi otomatis dari ekstensi)")
    ap.add_argument("--alpha", type=float, default=0.8,
                    help="[video] smoothing temporal 0-1 (default 0.8)")
    ap.add_argument("--skip", type=int, default=1,
                    help="[video] prediksi tiap N frame (default 1)")
    ap.add_argument("--fps", type=float, default=None,
                    help="[video] paksa FPS output (misal: 24, 25, 30)")
    ap.add_argument("--no-preview", action="store_true",
                    help="matikan jendela pratinjau")
    args = ap.parse_args()

    if args.input is None:
        menu_interaktif()
    else:
        jenis = args.mode or deteksi_jenis(args.input)
        if jenis is None:
            raise SystemExit("Jenis berkas tidak dikenali. Gunakan --mode foto atau --mode video.")

        keluaran = args.output or nama_keluaran(args.input, jenis)
        if jenis == "foto":
            proses_foto(args.input, keluaran, tampil=not args.no_preview)
        else:
            proses_video(args.input, keluaran,
                         tampil=not args.no_preview,
                         alpha=args.alpha, setiap_n=args.skip,
                         paksa_fps=args.fps)