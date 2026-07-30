import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

from tensorflow.keras.models import load_model

# ===========================
# KONFIGURASI
# ===========================

st.set_page_config(
    page_title="Prediksi Harga Emas GRU",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU")
st.caption("Model GRU dengan mempertimbangkan variabel USD dan Inflasi")

# ===========================
# LOAD MODEL
# ===========================

@st.cache_resource
def load_gru():
    model = load_model("gru_gold_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_gru()

# ===========================
# SIDEBAR
# ===========================

st.sidebar.title("Informasi Model")

st.sidebar.write("### Arsitektur")

st.sidebar.write("""
- GRU (64 Unit)
- Dropout (0.2)
- Dense (1)
- Optimizer Adam
- Loss MSE
""")

st.sidebar.write("---")

st.sidebar.write("### Dataset")

st.sidebar.write("""
Gold : Yahoo Finance

USD : Yahoo Finance

Inflasi : FRED
""")

# ===========================
# DOWNLOAD DATA
# ===========================

st.header("Download Data")

gold = yf.download(
    "GC=F",
    start="2001-01-01",
    end="2026-01-01",
    interval="1mo",
    auto_adjust=False
)

usd = yf.download(
    "IDR=X",
    start="2001-01-01",
    end="2026-01-01",
    interval="1mo",
    auto_adjust=False
)

gold.columns = gold.columns.get_level_values(0)
usd.columns = usd.columns.get_level_values(0)

gold = gold.reset_index()
usd = usd.reset_index()

gold = gold.rename(columns={
    "Open":"gold_open",
    "High":"gold_high",
    "Low":"gold_low",
    "Close":"gold_close"
})

usd = usd.rename(columns={
    "Open":"usd_open",
    "High":"usd_high",
    "Low":"usd_low",
    "Close":"usd_close"
})

gold = gold[[
    "Date",
    "gold_open",
    "gold_high",
    "gold_low",
    "gold_close"
]]

usd = usd[[
    "Date",
    "usd_open",
    "usd_high",
    "usd_low",
    "usd_close"
]]

df = gold.merge(
    usd,
    on="Date"
)

st.success("Data berhasil dimuat")

st.dataframe(df.tail())

# ===========================
# LOAD DATA INFLASI
# ===========================

inflation = pd.read_csv("inflation.csv")

inflation["Date"] = pd.to_datetime(inflation["Date"])

# ===========================
# MERGE DATA
# ===========================

df = df.merge(
    inflation,
    on="Date",
    how="inner"
)

df = df.sort_values("Date")

# ===========================
# FITUR MODEL
# ===========================

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

# ===========================
# NORMALISASI
# ===========================

scaled = scaler.transform(data)

# ===========================
# TIME STEP
# ===========================

TIME_STEPS = 24

last_sequence = scaled[-TIME_STEPS:]

X = np.expand_dims(
    last_sequence,
    axis=0
)

# ===========================
# PREDIKSI
# ===========================

pred = model.predict(
    X,
    verbose=0
)

target_index = features.index("gold_close")

dummy = np.zeros(
    (
        1,
        len(features)
    )
)

dummy[0, target_index] = pred[0][0]

prediction = scaler.inverse_transform(dummy)[0][target_index]

# ===========================
# NILAI TERAKHIR
# ===========================

last_price = df["gold_close"].iloc[-1]

change = prediction - last_price

percent = (change / last_price) * 100

# =====================================
# HASIL PREDIKSI
# =====================================

st.header("Hasil Prediksi")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Harga Emas Terakhir",
    f"${last_price:,.2f}"
)

col2.metric(
    "Prediksi Berikutnya",
    f"${prediction:,.2f}",
    delta=f"{change:,.2f}"
)

col3.metric(
    "Perubahan (%)",
    f"{percent:.2f}%"
)

st.divider()

# =====================================
# VISUALISASI
# =====================================

st.subheader("Grafik Harga Emas")

chart = df[
    [
        "Date",
        "gold_close"
    ]
].copy()

chart = chart.set_index("Date")

st.line_chart(chart)

# =====================================
# TAMPILKAN DATA
# =====================================

st.subheader("Dataset")

st.dataframe(df)

# =====================================
# DETAIL PREDIKSI
# =====================================

st.subheader("Ringkasan")

if prediction > last_price:

    st.success(
        f"""
Prediksi model menunjukkan harga emas bulan berikutnya diperkirakan naik.

Harga terakhir : {last_price:.2f}

Prediksi : {prediction:.2f}

Kenaikan : {change:.2f}
"""
    )

elif prediction < last_price:

    st.error(
        f"""
Prediksi model menunjukkan harga emas bulan berikutnya diperkirakan turun.

Harga terakhir : {last_price:.2f}

Prediksi : {prediction:.2f}

Penurunan : {abs(change):.2f}
"""
    )

else:

    st.info("Prediksi sama dengan harga terakhir.")

# =====================================
# DOWNLOAD DATA
# =====================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="Download Dataset",

    data=csv,

    file_name="gold_prediction.csv",

    mime="text/csv"

)

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
"""
Prediksi Harga Emas Menggunakan GRU

Model :
GRU(64) → Dropout(0.2) → Dense(1)

TensorFlow • Streamlit
"""
)
