import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Prediksi Harga Emas GRU",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU")
st.markdown(
"""
Aplikasi ini digunakan untuk memprediksi harga emas
berdasarkan model GRU yang telah dilatih.

Silakan pilih skenario prediksi kemudian pilih
periode yang ingin diprediksi.
"""
)

# =====================================================
# LOAD DATASET
# =====================================================

DATA_PATH = "Dataset_Fix.csv"

df = pd.read_csv(DATA_PATH)

# ubah kolom tanggal
df["date"] = pd.to_datetime(df["date"])

# urutkan
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# DAFTAR MODEL
# =====================================================

MODEL_INFO = {

    "Gold Only": {

        "model":"saved_models/gold_only.keras",
        "scaler":"saved_models/gold_only_scaler.pkl",
        "feature":"saved_models/gold_only_features.json"

    },

    "Gold + USD": {

        "model":"saved_models/gold_usd.keras",
        "scaler":"saved_models/gold_usd_scaler.pkl",
        "feature":"saved_models/gold_usd_features.json"

    },

    "Gold + Inflation": {

        "model":"saved_models/gold_inflasi.keras",
        "scaler":"saved_models/gold_inflasi_scaler.pkl",
        "feature":"saved_models/gold_inflasi_features.json"

    },

    "Gold + USD + Inflation": {

        "model":"saved_models/gold_usd_inflasi.keras",
        "scaler":"saved_models/gold_usd_inflasi_scaler.pkl",
        "feature":"saved_models/gold_usd_inflasi_features.json"

    }

}

TIME_STEPS = 12

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_saved_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {model_path}")
        st.exception(e)
        st.stop()


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

st.sidebar.title("⚙️ Pengaturan")

scenario = st.sidebar.selectbox(
    "Pilih Skenario",
    list(MODEL_INFO.keys())
)

info = MODEL_INFO[scenario]

model = load_saved_model(info["model"])
scaler = load_scaler(info["scaler"])
features = load_features(info["feature"])

st.sidebar.success("Model berhasil dimuat")

# =====================================================
# PILIH PERIODE PREDIKSI
# =====================================================

available_dates = df["date"].dt.strftime("%Y-%m").tolist()

selected_period = st.selectbox(
    "Pilih Periode yang Akan Diprediksi",
    available_dates[TIME_STEPS:]
)

selected_index = df[
    df["date"].dt.strftime("%Y-%m") == selected_period
].index[0]

# =====================================================
# CEK DATA
# =====================================================

if selected_index < TIME_STEPS:

    st.error("Data sebelumnya belum mencukupi.")
    st.stop()

# =====================================================
# AMBIL 12 BULAN SEBELUMNYA
# =====================================================

history = df.iloc[
    selected_index-TIME_STEPS:selected_index
].copy()

actual_row = df.iloc[selected_index]

st.subheader("12 Bulan Data Sebelumnya")

st.dataframe(history)

# =====================================================
# AMBIL FITUR
# =====================================================

X = history[features].values

# scaling

X_scaled = scaler.transform(X)

# ubah menjadi bentuk
# (1,12,jumlah_fitur)

X_input = np.expand_dims(
    X_scaled,
    axis=0
)

st.success("Data siap diprediksi.")
# =====================================================
# TOMBOL PREDIKSI
# =====================================================

if st.button("🔮 Prediksi Harga Emas", use_container_width=True):

    with st.spinner("Melakukan prediksi..."):

        # ===============================
        # PREDIKSI
        # ===============================

        prediction_scaled = model.predict(X_input, verbose=0)

        # ===============================
        # INVERSE TRANSFORM
        # ===============================

        target_index = features.index("gold_close")

        dummy = np.zeros((1, len(features)))
        dummy[:, target_index] = prediction_scaled.flatten()

        prediction = scaler.inverse_transform(dummy)[0][target_index]

        # ===============================
        # AKTUAL
        # ===============================

        actual = actual_row["gold_close"]

        # ===============================
        # ERROR
        # ===============================

        mae = abs(actual - prediction)

        mape = (mae / actual) * 100

        # ===============================
        # HASIL
        # ===============================

        st.success("Prediksi berhasil dilakukan.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Harga Aktual",
                f"{actual:.2f}"
            )

        with c2:
            st.metric(
                "Harga Prediksi",
                f"{prediction:.2f}"
            )

        with c3:
            st.metric(
                "Selisih",
                f"{mae:.2f}"
            )

        st.metric(
            "MAPE",
            f"{mape:.3f}%"
        )

        # ===============================
        # GRAFIK
        # ===============================

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Aktual", "Prediksi"],
                y=[actual, prediction],
                text=[f"{actual:.2f}", f"{prediction:.2f}"],
                textposition="auto",
                name="Harga"
            )
        )

        fig.update_layout(
            title="Perbandingan Harga Aktual dan Prediksi",
            yaxis_title="Harga Emas",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # INFORMASI TAMBAHAN
        # ===============================

        st.subheader("Informasi Prediksi")

        info_df = pd.DataFrame({
            "Periode":[selected_period],
            "Skenario":[scenario],
            "Harga Aktual":[actual],
            "Harga Prediksi":[prediction],
            "MAE":[mae],
            "MAPE (%)":[mape]
        })

        st.dataframe(info_df, use_container_width=True)
        # =====================================================
# RINGKASAN HASIL
# =====================================================

st.markdown("---")

st.subheader("📋 Ringkasan Prediksi")

hasil = pd.DataFrame({
    "Periode": [selected_period],
    "Skenario": [scenario],
    "Harga Aktual": [round(actual,2)],
    "Harga Prediksi": [round(prediction,2)],
    "MAE": [round(mae,2)],
    "MAPE (%)": [round(mape,3)]
})

st.dataframe(
    hasil,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# RIWAYAT 12 BULAN
# =====================================================

st.subheader("📈 Riwayat Harga 12 Bulan")

history_chart = history.copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history_chart["date"],
        y=history_chart["gold_close"],
        mode="lines+markers",
        name="Gold Close"
    )
)

fig.update_layout(
    title="Harga Gold Close 12 Bulan Sebelumnya",
    xaxis_title="Tanggal",
    yaxis_title="Harga",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PERBANDINGAN AKTUAL VS PREDIKSI
# =====================================================

st.subheader("📊 Aktual vs Prediksi")

compare = pd.DataFrame({
    "Kategori":["Aktual","Prediksi"],
    "Harga":[actual,prediction]
})

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=compare["Kategori"],
        y=compare["Harga"],
        text=np.round(compare["Harga"],2),
        textposition="outside"
    )
)

fig2.update_layout(
    title="Perbandingan Aktual dan Prediksi",
    height=500,
    yaxis_title="Harga Gold"
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# DOWNLOAD CSV
# =====================================================

csv = hasil.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Hasil Prediksi",
    csv,
    "hasil_prediksi.csv",
    "text/csv"
)
