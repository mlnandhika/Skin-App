from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
ONNX_PATH = BASE_DIR / "models" / "onnx" / "model.onnx"
LABEL_PATH = BASE_DIR / "models" / "onnx" / "label.txt"
IMG_SIZE = 224

st.set_page_config(
    page_title="Skin Check",
    page_icon=":material/dermatology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Informasi per kelas (urutan label mengikuti label.txt hasil training)
KONDISI = {
    "acne": {
        "nama": "Jerawat (Acne)",
        "ringkas": "Peradangan pada folikel rambut dan kelenjar minyak kulit.",
        "ciri": "Komedo, benjolan merah, atau bintik bernanah, umumnya di wajah, dada, dan punggung.",
        "saran": "Cuci wajah dengan pembersih lembut, hindari memencet jerawat, dan pilih produk berlabel non-komedogenik. Temui dokter jika jerawat meradang berat, nyeri, atau meninggalkan bekas.",
    },
    "eksim": {
        "nama": "Eksim (Dermatitis)",
        "ringkas": "Kondisi kulit kronis yang membuat kulit kering, gatal, dan mudah meradang.",
        "ciri": "Bercak merah, kulit bersisik atau pecah-pecah, gatal terutama malam hari, dan sering kambuh.",
        "saran": "Gunakan pelembap secara rutin, kenali dan hindari pemicu (sabun keras, keringat, alergen), serta jangan menggaruk. Obat oles steroid sebaiknya hanya dipakai atas anjuran dokter.",
    },
    "herpes": {
        "nama": "Herpes",
        "ringkas": "Infeksi akibat virus herpes yang ditandai lepuh berisi cairan.",
        "ciri": "Lepuh berkelompok di atas kulit kemerahan, terasa perih, panas, atau nyeri sebelum lepuh muncul.",
        "saran": "Periksakan ke dokter sesegera mungkin karena obat antivirus paling efektif bila dimulai sejak awal. Hindari menyentuh lepuh dan kontak langsung dengan orang lain selama masih ada luka.",
    },
    "panu": {
        "nama": "Panu (Tinea versicolor)",
        "ringkas": "Infeksi jamur superfisial yang mengubah warna kulit pada area tertentu.",
        "ciri": "Bercak putih atau kecokelatan dengan sisik halus, gatal ringan, sering di punggung, dada, dan lengan atas.",
        "saran": "Jaga kulit tetap bersih dan kering, jangan berbagi handuk atau pakaian, dan gunakan antijamur topikal sesuai anjuran tenaga kesehatan. Warna kulit bisa baru kembali normal beberapa minggu setelah jamur teratasi.",
    },
    "rosacea": {
        "nama": "Rosacea",
        "ringkas": "Gangguan kulit wajah kronis yang menyebabkan kemerahan menetap.",
        "ciri": "Kemerahan di pipi, hidung, atau dahi, pembuluh darah halus tampak, kadang disertai benjolan mirip jerawat.",
        "saran": "Pakai tabir surya setiap hari, catat dan hindari pemicu seperti panas, alkohol, atau makanan pedas, lalu konsultasikan ke dokter kulit untuk terapi jangka panjang.",
    },
}

# Hasil evaluasi pada test set (dari notebook)
LAPORAN = pd.DataFrame(
    {
        "Kelas": ["Acne", "Eksim", "Herpes", "Panu", "Rosacea"],
        "Precision": [0.95, 0.81, 0.92, 0.88, 0.95],
        "Recall": [0.89, 0.87, 0.98, 0.96, 0.80],
        "F1-score": [0.92, 0.84, 0.95, 0.91, 0.87],
        "Jumlah data uji": [45, 45, 45, 45, 45],
    }
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    import onnxruntime as ort

    sesi = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    label = [baris.strip() for baris in LABEL_PATH.read_text().splitlines() if baris.strip()]
    return sesi, label


def praproses(gambar: Image.Image) -> np.ndarray:
    # Preprocessing MobileNetV3 sudah ada di dalam model,
    # jadi input cukup RGB 224x224 dengan nilai piksel 0-255 (float32).
    gambar = ImageOps.exif_transpose(gambar).convert("RGB")
    gambar = gambar.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.expand_dims(np.asarray(gambar, dtype=np.float32), axis=0)


def prediksi(gambar: Image.Image) -> pd.DataFrame:
    sesi, label = muat_model()
    nama_input = sesi.get_inputs()[0].name
    probabilitas = sesi.run(None, {nama_input: praproses(gambar)})[0][0]
    hasil = pd.DataFrame({"kode": label, "prob": probabilitas * 100})
    return hasil.sort_values("prob", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Komponen tampilan
# --------------------------------------------------------------------------
def tampilkan_hasil(hasil: pd.DataFrame, batas: int):
    teratas = hasil.iloc[0]
    kedua = hasil.iloc[1]
    info = KONDISI.get(teratas["kode"], {})
    selisih = teratas["prob"] - kedua["prob"]

    with st.container(border=True):
        st.caption("Prediksi teratas")
        st.subheader(info.get("nama", teratas["kode"]), anchor=False)

        k1, k2 = st.columns(2)
        k1.metric("Tingkat keyakinan", f"{teratas['prob']:.1f}%")
        k2.metric(
            "Selisih dengan kandidat kedua",
            f"{selisih:.1f} poin",
            help=f"Kandidat kedua: {KONDISI.get(kedua['kode'], {}).get('nama', kedua['kode'])}",
        )
        st.progress(min(float(teratas["prob"]) / 100, 1.0))

    if teratas["prob"] < batas:
        st.warning(
            f"Keyakinan model di bawah {batas}%. Coba foto ulang dengan pencahayaan lebih baik "
            "atau lebih dekat ke area kulit, dan jangan jadikan hasil ini sebagai acuan.",
            icon=":material/warning:",
        )
    elif selisih < 15:
        st.info(
            "Dua kandidat teratas nilainya berdekatan, sehingga gejalanya mungkin mirip. "
            "Pemeriksaan langsung oleh dokter akan jauh lebih akurat.",
            icon=":material/info:",
        )

    st.markdown("**Probabilitas semua kelas**")
    tabel = pd.DataFrame(
        {
            "Kondisi": [KONDISI.get(k, {}).get("nama", k) for k in hasil["kode"]],
            "Probabilitas": hasil["prob"].round(2),
        }
    )
    st.dataframe(
        tabel,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilitas": st.column_config.ProgressColumn(
                "Probabilitas", format="%.1f%%", min_value=0, max_value=100
            )
        },
    )

    if info:
        with st.expander(f"Mengenal {info['nama']}", expanded=True, icon=":material/menu_book:"):
            st.write(info["ringkas"])
            st.markdown(f"**Ciri yang umum:** {info['ciri']}")
            st.markdown(f"**Langkah awal:** {info['saran']}")


def halaman_analisis(batas: int):
    kiri, kanan = st.columns([5, 6], gap="large")

    with kiri:
        sumber = st.segmented_control(
            "Sumber gambar",
            ["Unggah file", "Kamera"],
            default="Unggah file",
            label_visibility="collapsed",
        )

        berkas = None
        if sumber == "Kamera":
            berkas = st.camera_input("Ambil foto area kulit", label_visibility="collapsed")
        else:
            berkas = st.file_uploader(
                "Pilih foto area kulit",
                type=["jpg", "jpeg", "png", "webp"],
                help="Maksimal 10 MB. Gambar tidak disimpan oleh aplikasi.",
            )

        if berkas is not None:
            gambar = Image.open(berkas)
            st.image(ImageOps.exif_transpose(gambar), caption=berkas.name if hasattr(berkas, "name") else None, width="stretch")

        with st.expander("Tips pengambilan foto", icon=":material/photo_camera:"):
            st.markdown(
                "- Gunakan cahaya alami atau lampu terang, hindari bayangan.\n"
                "- Dekatkan kamera sampai area kulit memenuhi sebagian besar frame.\n"
                "- Pastikan fokus tajam dan tidak ada filter.\n"
                "- Satu foto fokus pada satu area."
            )

    with kanan:
        if berkas is None:
            with st.container(border=True):
                st.markdown("#### Belum Ada Gambar")
                st.write(
                    "Unggah atau ambil foto di panel kiri. "
                    "Hasil analisis akan muncul di sini beberapa detik setelahnya."
                )
        else:
            try:
                with st.spinner("Menganalisis gambar..."):
                    hasil = prediksi(gambar)
                tampilkan_hasil(hasil, batas)
            except FileNotFoundError:
                st.error(
                    "File model tidak ditemukan. Pastikan folder `model/onnx/` berisi "
                    "`model.onnx` dan `label.txt`.",
                    icon=":material/error:",
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Gambar tidak dapat diproses: {e}", icon=":material/error:")


def halaman_model():
    st.markdown("#### Performa pada data uji")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", "89,78%")
    m2.metric("Macro F1", "0,90")
    m3.metric("Total gambar", "1.494")
    m4.metric("Jumlah kelas", "5")

    st.dataframe(
        LAPORAN,
        hide_index=True,
        width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
            "F1-score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
    )

    st.divider()
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Cara Kerja")
        st.markdown(
            "- **Arsitektur:** MobileNetV3 Small (transfer learning dari ImageNet)\n"
            "- **Pelatihan:** Dua tahap, backbone dibekukan lalu fine-tuning 20 layer terakhir\n"
            "- **Input:** Gambar RGB 224 × 224 piksel\n"
            "- **Pembagian data:** 70% latih, 15% validasi, 15% uji (stratified)\n"
            "- **Format inferensi:** ONNX Runtime di CPU"
        )
    with b:
        st.markdown("#### Keterbatasan")
        st.markdown(
            "- Model hanya mengenal lima kondisi yaitu acne, eksim, herpes, panu, dan rosacea.\n"
            "- Kulit sehat atau penyakit lain tetap akan dipaksa masuk ke salah satu kelas.\n"
            "- Hasil dipengaruhi kualitas foto, pencahayaan, dan warna kulit.\n"
            "- Data latih relatif kecil, sehingga hasil perlu dibaca sebagai indikasi awal."
        )


# --------------------------------------------------------------------------
# Layout utama
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Skin Check", anchor=False)
    st.caption("Identifikasi awal kondisi kulit secara praktis")
    st.divider()

    batas = st.slider(
        "**Batas Keyakinan Minimum**",
        min_value=30,
        max_value=90,
        value=60,
        step=5,
        format="%d%%",
        help="Jika keyakinan model di bawah angka ini, aplikasi akan menampilkan peringatan.",
    )

    st.divider()
    st.markdown("**Kondisi yang Dikenali:**")
    for k in KONDISI.values():
        st.markdown(f"- {k['nama']}")

st.title("Analisis Kondisi Kulit", anchor=False)
st.caption("Alat bantu skrining berbasis deep learning, bukan pengganti diagnosis dokter. " \
"Hasil analisis ini berfungsi sebagai informasi awal dan disarankan untuk dikonsultasikan lebih lanjut dengan tenaga medis profesional. " \
"Seluruh hasil yang ditampilkan bersifat sebagai referensi pendukung dan tidak boleh digunakan sebagai satu-satunya dasar pengambilan keputusan medis.")

tab_analisis, tab_model = st.tabs(["Analisis", "Tentang model"])
with tab_analisis:
    halaman_analisis(batas)
with tab_model:
    halaman_model()

st.divider()
st.caption(
    "Aplikasi ini hanya untuk tujuan edukasi dan skrining awal. Untuk diagnosis dan pengobatan, "
    "konsultasikan dengan dokter atau dokter spesialis kulit."
)
