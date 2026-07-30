import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Prediksi Harga Emas Menggunakan GRU",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#0f172a;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

.title-box{
    background:#172554;
    border-radius:18px;
    padding:30px;
    text-align:center;
    color:white;
    box-shadow:0px 0px 20px rgba(0,0,0,.35);
}

.card{

    background:#1e293b;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 0px 15px rgba(0,0,0,.25);

}

.result-card{

    background:#172554;

    color:white;

    text-align:center;

    border-radius:15px;

    padding:35px;

    box-shadow:0px 0px 25px rgba(37,99,235,.35);

}

.stButton>button{

    width:100%;

    background:#2563eb;

    color:white;

    border:none;

    border-radius:12px;

    height:55px;

    font-size:18px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#1d4ed8;

    color:white;

}

hr{

    margin-top:35px;

    margin-bottom:35px;

}

</style>
""",unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_gru():

    model=load_model("gru_gold_model.keras")

    scaler=joblib.load("scaler.pkl")

    return model,scaler


model,scaler=load_gru()

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""

<div class="title-box">

<h1>📈 Prediksi Harga Emas Menggunakan GRU</h1>

<h3>Dengan Mempertimbangkan Variabel USD dan Inflasi</h3>

<p>

Upload dataset bulanan kemudian tekan tombol prediksi
untuk memprediksi harga emas bulan berikutnya menggunakan
model GRU.

</p>

</div>

""",unsafe_allow_html=True)

st.write("")

# ==========================================================
# PILIH DATASET
# ==========================================================

left,right=st.columns([2,1])

with left:

    option=st.radio(

        "Sumber Dataset",

        [

            "Gunakan Dataset Bawaan",

            "Upload Dataset Baru"

        ],

        horizontal=True

    )

with right:

    st.info("Model : GRU")

# ==========================================================
# LOAD DATA
# ==========================================================

if option=="Gunakan Dataset Bawaan":

    df=pd.read_csv("Dataset_Fix.csv")

else:

    uploaded=st.file_uploader(

        "Upload Dataset CSV",

        type=["csv"]

    )

    if uploaded is None:

        st.warning("Silakan upload dataset terlebih dahulu.")

        st.stop()

    df=pd.read_csv(uploaded)

# ==========================================================
# PREPROCESS
# ==========================================================

df["date"]=pd.to_datetime(df["date"])

df=df.sort_values("date")

df.reset_index(drop=True,inplace=True)

st.write("")

st.success("✅ Dataset berhasil dimuat")

col1,col2,col3=st.columns(3)

with col1:

    st.metric(

        "Jumlah Data",

        len(df)

    )

with col2:

    st.metric(

        "Tanggal Awal",

        df["date"].min().strftime("%Y-%m")

    )

with col3:

    st.metric(

        "Tanggal Akhir",

        df["date"].max().strftime("%Y-%m")

    )

st.write("")

st.subheader("Preview Dataset")

st.dataframe(

    df.head(),

    use_container_width=True

)

# ==========================================================
# FEATURE YANG DIGUNAKAN MODEL
# ==========================================================

FEATURES = [

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

TARGET = "gold_close"

TIME_STEP = 24

# ==========================================================
# AMBIL FEATURE
# ==========================================================

data = df[FEATURES].copy()

scaled = scaler.transform(data)

# ==========================================================
# TOMBOL PREDIKSI
# ==========================================================

st.write("")

st.subheader("🤖 Prediksi")

predict = st.button("🔮 Prediksi Harga Bulan Berikutnya")

# ==========================================================
# PROSES PREDIKSI
# ==========================================================

if predict:

    with st.spinner("Model GRU sedang melakukan prediksi..."):

        sequence = scaled[-TIME_STEP:]

        X = np.array(sequence)

        X = np.expand_dims(X, axis=0)

        pred_scaled = model.predict(

            X,

            verbose=0

        )

# ==========================================================
# INVERSE TRANSFORM
# ==========================================================

        target_index = FEATURES.index(TARGET)

        dummy = np.zeros(

            (

                1,

                len(FEATURES)

            )

        )

        dummy[0, target_index] = pred_scaled[0][0]

        pred = scaler.inverse_transform(dummy)

        prediction = pred[0, target_index]

# ==========================================================
# HARGA TERAKHIR
# ==========================================================

        actual = df[TARGET].iloc[-1]

        diff = prediction - actual

        pct = (diff / actual) * 100

# ==========================================================
# HASIL
# ==========================================================

    st.write("")

    st.subheader("📊 Hasil Prediksi")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""

        <div class="card">

        <h4>Harga Aktual Terakhir</h4>

        </div>

        """, unsafe_allow_html=True)

        st.metric(

            "",

            f"${actual:,.2f}"

        )

    with c2:

        st.markdown("""

        <div class="card">

        <h4>Prediksi Bulan Berikutnya</h4>

        </div>

        """, unsafe_allow_html=True)

        st.metric(

            "",

            f"${prediction:,.2f}",

            f"{diff:,.2f}"

        )

    with c3:

        st.markdown("""

        <div class="card">

        <h4>Perubahan (%)</h4>

        </div>

        """, unsafe_allow_html=True)

        st.metric(

            "",

            f"{pct:.2f}%"

        )

# ==========================================================
# KARTU HASIL
# ==========================================================

    st.write("")

    if prediction > actual:

        warna = "#16a34a"

        status = "📈 Harga diprediksi NAIK"

    elif prediction < actual:

        warna = "#dc2626"

        status = "📉 Harga diprediksi TURUN"

    else:

        warna = "#2563eb"

        status = "➖ Harga diprediksi TETAP"

    st.markdown(

        f"""

        <div style="

        background:{warna};

        padding:30px;

        border-radius:15px;

        text-align:center;

        color:white;

        ">

        <h2>{status}</h2>

        <h1>${prediction:,.2f}</h1>

        <h3>Selisih : {diff:,.2f}</h3>

        <h3>Persentase : {pct:.2f}%</h3>

        </div>

        """,

        unsafe_allow_html=True

    )
    # ==========================================================
# GRAFIK PREDIKSI
# ==========================================================

    st.write("")

    st.subheader("📈 Visualisasi Prediksi")

    chart = df[["date", TARGET]].copy()

    pred_date = chart["date"].iloc[-1] + pd.DateOffset(months=1)

    pred_df = pd.DataFrame({

        "date": [pred_date],

        TARGET: [prediction]

    })

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=chart["date"],

            y=chart[TARGET],

            mode="lines+markers",

            name="Harga Aktual"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=pred_df["date"],

            y=pred_df[TARGET],

            mode="markers",

            marker=dict(

                size=15,

                color="red"

            ),

            name="Prediksi"

        )

    )

    fig.update_layout(

        title="Grafik Harga Emas dan Prediksi",

        xaxis_title="Tanggal",

        yaxis_title="Harga",

        template="plotly_white",

        height=550

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# RINGKASAN
# ==========================================================

    st.write("")

    st.subheader("📋 Ringkasan Prediksi")

    summary = pd.DataFrame({

        "Harga Aktual Terakhir": [round(actual, 2)],

        "Prediksi": [round(prediction, 2)],

        "Selisih": [round(diff, 2)],

        "Perubahan (%)": [round(pct, 2)]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

# ==========================================================
# DOWNLOAD
# ==========================================================

    csv = summary.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        "📥 Download Hasil Prediksi",

        csv,

        "hasil_prediksi.csv",

        "text/csv"

    )

# ==========================================================
# INFORMASI MODEL
# ==========================================================

st.write("")

with st.expander("ℹ️ Informasi Model"):

    st.markdown("""

### Model

GRU (Gated Recurrent Unit)

### Input Feature

- Gold Open
- Gold High
- Gold Low
- Gold Close
- USD Open
- USD High
- USD Low
- USD Close
- Inflation Rate

### Target

Gold Close

### Time Step

24 Bulan

### Optimizer

Adam

### Loss Function

Mean Squared Error (MSE)

### Normalisasi

MinMaxScaler

""")

# ==========================================================
# FOOTER
# ==========================================================

st.write("")

st.markdown("---")

st.markdown(

"""

<div style="text-align:center">

<b>Prediksi Harga Emas Menggunakan Model GRU</b>

<br>

Dengan Mempertimbangkan Variabel USD dan Inflasi

<br><br>

Universitas AMIKOM Yogyakarta

</div>

""",

unsafe_allow_html=True

)
