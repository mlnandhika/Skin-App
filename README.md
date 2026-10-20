# Skin App

Aplikasi web untuk klasifikasi awal kondisi kulit dari foto, dibangun dengan Streamlit dan model deep learning berbasis MobileNetV3 Small.

## Kondisi yang Dikenali

| Kelas | Keterangan |
| --- | --- |
| `acne` | Jerawat |
| `eksim` | Eksim (dermatitis) |
| `herpes` | Herpes |
| `panu` | Panu (tinea versicolor) |
| `rosacea` | Rosacea |

## Struktur Proyek

```
Skin-App/
├── app.py                  # Aplikasi Streamlit
├── requirements.txt        # Dependensi
├── README.md
├── .streamlit/
│   └── config.toml         # Tema dan konfigurasi Streamlit
├── demo/                   
│   ├── acne/               
│   │   ├── ac10_JPG.rf...  
│   │   ├── ac16_jpg.rf...  
│   │   └── ac17_jpg.rf...  
│   ├── eksim/              
│   │   ├── ek15_JPG.rf...  
│   │   ├── ek16_JPG.rf...  
│   │   └── ek100_jpg.rf... 
│   ├── herpes/             
│   │   ├── he15_jpg.rf...  
│   │   ├── he18_jpg.rf...  
│   │   └── he100_jpg.rf... 
│   ├── panu/               
│   │   ├── pa11_jpg.rf...  
│   │   ├── pa12_jpg.rf...  
│   │   └── pa100_jpeg.rf...
│   └── rosacea/            
│       ├── ro12_jpg.rf...  
│       ├── ro16_jpg.rf...  
│       └── ro18_jpg.rf...  
├── model/
│   └── onnx/
│       ├── model.onnx      # Model hasil ekspor
│       └── label.txt       # Urutan label kelas
└── notebook/
    └── classification-of-skin-diseases.ipynb
```

## Instalasi

1. **Buat virtual environment** (disarankan, Python 3.10 atau lebih baru).

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Instal dependensi.**

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi.**

   ```bash
   streamlit run app.py
   ```

   Aplikasi terbuka di `http://localhost:8501`.

## Deployment

Web juga dapat diakses melalui https://skin-app-diseases.streamlit.app/

## Dataset

**Dataset:** [Classification of Skin Diseases (Kaggle)](https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases), 1.494 gambar dalam 5 kelas.