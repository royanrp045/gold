import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediksi Harga Emas GRU",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU")

st.write(
"""
Aplikasi ini menggunakan model GRU
untuk memprediksi harga emas
berdasarkan model yang telah dilatih.
"""
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("Dataset_Fix.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

TIME_STEPS = 12

# =====================================================
# MODEL
# =====================================================

MODEL_INFO = {

    "Gold Only":{

        "model":"saved_models/gold_only.keras",
        "scaler":"saved_models/gold_only_scaler.pkl",
        "feature":"saved_models/gold_only_features.json"

    },

    "Gold + USD":{

        "model":"saved_models/gold_usd.keras",
        "scaler":"saved_models/gold_usd_scaler.pkl",
        "feature":"saved_models/gold_usd_features.json"

    },

    "Gold + Inflation":{

        "model":"saved_models/gold_inflasi.keras",
        "scaler":"saved_models/gold_inflasi_scaler.pkl",
        "feature":"saved_models/gold_inflasi_features.json"

    },

    "Gold + USD + Inflation":{

        "model":"saved_models/gold_usd_inflasi.keras",
        "scaler":"saved_models/gold_usd_inflasi_scaler.pkl",
        "feature":"saved_models/gold_usd_inflasi_features.json"

    }

}

# =====================================================
# LOAD FILE
# =====================================================

@st.cache_resource
def load_gru(path):

    return load_model(path, compile=False)


@st.cache_resource
def load_scaler(path):

    return joblib.load(path)


@st.cache_resource
def load_features(path):

    with open(path,"r") as f:

        return json.load(f)
        # =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Pengaturan")

scenario = st.sidebar.selectbox(
    "Pilih Skenario",
    list(MODEL_INFO.keys())
)

info = MODEL_INFO[scenario]

# =====================================================
# LOAD MODEL
# =====================================================

model = load_gru(info["model"])
scaler = load_scaler(info["scaler"])
features = load_features(info["feature"])

st.sidebar.success("Model berhasil dimuat")

# =====================================================
# PILIH PERIODE
# =====================================================

available_dates = df["date"].dt.strftime("%Y-%m").tolist()

selected_period = st.selectbox(
    "Pilih Periode Prediksi",
    available_dates[TIME_STEPS:]
)

selected_index = df[
    df["date"].dt.strftime("%Y-%m") == selected_period
].index[0]

# =====================================================
# AMBIL DATA 12 BULAN
# =====================================================

history = df.iloc[
    selected_index - TIME_STEPS : selected_index
].copy()

# =====================================================
# AMBIL FITUR SESUAI MODEL
# =====================================================

X = history[features].values

# =====================================================
# NORMALISASI
# =====================================================

X_scaled = scaler.transform(X)

# =====================================================
# UBAH MENJADI SHAPE
# (1,12,n_features)
# =====================================================

X_input = np.expand_dims(
    X_scaled,
    axis=0
)
# =====================================================
# PREDIKSI
# =====================================================

st.markdown("---")

if st.button("🔮 Prediksi Harga Emas", use_container_width=True):

    with st.spinner("Sedang melakukan prediksi..."):

        # Prediksi menggunakan model GRU
        prediction_scaled = model.predict(X_input, verbose=0)

        # Posisi target (gold_close)
        target_index = features.index("gold_close")

        # Dummy array untuk inverse transform
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[:, target_index] = prediction_scaled.flatten()

        # Kembalikan ke skala asli
        prediction = scaler.inverse_transform(dummy)[0][target_index]

    st.success("✅ Prediksi Berhasil")

    st.markdown("### Hasil Prediksi")

    st.metric(
        label="Harga Prediksi Emas",
        value=f"{prediction:.2f} USD"
    )
