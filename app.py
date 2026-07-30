import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi Harga Emas Menggunakan GRU",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU")
st.markdown(
    """
Aplikasi ini digunakan untuk melakukan prediksi harga emas
menggunakan model **GRU (Gated Recurrent Unit)** dengan
mempertimbangkan variabel **USD** dan **Inflasi**.
"""
)

# ==========================================================
# LOAD MODEL DAN SCALER
# ==========================================================

@st.cache_resource
def load_model_and_scaler():

    model = load_model("gru_gold_model.keras")

    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_model_and_scaler()

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("Dataset_Fix.csv")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df = df.reset_index(drop=True)

    return df


df = load_dataset()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Informasi Model")

st.sidebar.success("Model berhasil dimuat")

st.sidebar.markdown(
"""
### Model

- GRU
- 64 Unit
- Dropout 0.2
- Dense 1
- Optimizer Adam

---

### Dataset

Dataset_Fix.csv

---

### Input Feature

- gold_open
- gold_high
- gold_low
- gold_close
- usd_open
- usd_high
- usd_low
- usd_close
- inflation_rate

---

### Target

gold_close
"""
)

# ==========================================================
# PREVIEW DATASET
# ==========================================================

st.subheader("Preview Dataset")

st.dataframe(df.head(10))

# ==========================================================
# FEATURE YANG DIGUNAKAN MODEL
# ==========================================================

features = [

    "gold_open",
    "gold_high",
    "gold_low",
    "gold_close",

    "usd_open",
    "usd_high",
    "usd_low",
    "usd_close",

    "inflation_rate"

]

data = df[features]

# ==========================================================
# NORMALISASI DATA
# ==========================================================

scaled_data = scaler.transform(data)

# ==========================================================
# TIME STEP
# ==========================================================

TIME_STEPS = 24

last_sequence = scaled_data[-TIME_STEPS:]

X = np.expand_dims(last_sequence, axis=0)

# ==========================================================
# PROSES PREDIKSI
# ==========================================================

prediction_scaled = model.predict(
    X,
    verbose=0
)

target_index = features.index("gold_close")

dummy_prediction = np.zeros(
    (
        1,
        scaled_data.shape[1]
    )
)

dummy_prediction[0, target_index] = prediction_scaled[0][0]

prediction = scaler.inverse_transform(
    dummy_prediction
)[0][target_index]

# ==========================================================
# HARGA TERAKHIR
# ==========================================================

last_price = df["gold_close"].iloc[-1]

change = prediction - last_price

percent = (change / last_price) * 100

# ==========================================================
# HASIL PREDIKSI
# ==========================================================

st.header("Hasil Prediksi")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        label="Harga Emas Terakhir",
        value=f"${last_price:,.2f}"
    )

with col2:

    st.metric(
        label="Prediksi Bulan Berikutnya",
        value=f"${prediction:,.2f}",
        delta=f"{change:,.2f}"
    )

with col3:

    st.metric(
        label="Perubahan (%)",
        value=f"{percent:.2f}%"
    )

st.divider()

# ==========================================================
# RINGKASAN
# ==========================================================

st.subheader("Ringkasan Prediksi")

if prediction > last_price:

    st.success(
        f"""
Model memprediksi harga emas akan mengalami kenaikan.

Harga terakhir : {last_price:.2f}

Prediksi : {prediction:.2f}

Kenaikan : {change:.2f}
"""
    )

elif prediction < last_price:

    st.error(
        f"""
Model memprediksi harga emas akan mengalami penurunan.

Harga terakhir : {last_price:.2f}

Prediksi : {prediction:.2f}

Penurunan : {abs(change):.2f}
"""
    )

else:

    st.info(
        "Prediksi sama dengan harga terakhir."
    )

st.divider()

# ==========================================================
# INFORMASI DATA TERBARU
# ==========================================================

st.subheader("Data Terbaru")

st.dataframe(df.tail(10))

# ==========================================================
# GRAFIK HARGA EMAS
# ==========================================================

st.subheader("Grafik Harga Emas")

chart = df[
    [
        "date",
        "gold_close"
    ]
].copy()

chart = chart.set_index("date")

st.line_chart(chart)

# ==========================================================
# STATISTIK DATASET
# ==========================================================

st.subheader("Statistik Dataset")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Jumlah Data",
        len(df)
    )

with col2:

    st.metric(
        "Harga Tertinggi",
        f"${df['gold_close'].max():,.2f}"
    )

with col3:

    st.metric(
        "Harga Terendah",
        f"${df['gold_close'].min():,.2f}"
    )

# ==========================================================
# DATASET LENGKAP
# ==========================================================

st.subheader("Dataset Lengkap")

st.dataframe(
    df,
    use_container_width=True
)

# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="Dataset_Fix.csv",
    mime="text/csv"
)

# ==========================================================
# INFORMASI FITUR
# ==========================================================

with st.expander("Fitur yang Digunakan Model"):

    st.write(features)

# ==========================================================
# DESKRIPSI MODEL
# ==========================================================

with st.expander("Tentang Model"):

    st.markdown(
        """
Model menggunakan **Gated Recurrent Unit (GRU)**.

Input:
- Gold Open
- Gold High
- Gold Low
- Gold Close
- USD Open
- USD High
- USD Low
- USD Close
- Inflation Rate

Target:
- Gold Close

Normalisasi:
- MinMaxScaler

Optimizer:
- Adam

Loss Function:
- Mean Squared Error (MSE)
"""
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
"""
Prediksi Harga Emas Menggunakan Model GRU
dengan Mempertimbangkan Variabel USD dan Inflasi

Universitas AMIKOM Yogyakarta
"""
)
