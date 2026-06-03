# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stroke Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Animated gradient header */
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid #1e4a6e;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,191,255,0.06) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.main-header h1 { color: white; margin: 0; font-size: 28px; font-weight: 700; position: relative; z-index: 1; }
.main-header p  { color: #90b8d4; margin: 6px 0 0 0; font-size: 14px; position: relative; z-index: 1; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1a2744 0%, #0f1f3d 100%);
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #243b6b;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-align: center;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,191,255,0.15); }
.kpi-value { color: white; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-label { color: #90caf9; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 12px; margin-top: 4px; }
.kpi-red   .kpi-value { color: #FF4B4B; }
.kpi-blue  .kpi-value { color: #00BFFF; }
.kpi-green .kpi-value { color: #00e676; }
.kpi-orange .kpi-value { color: #FFA500; }

/* Info cards */
.info-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #00BFFF;
    transition: border-color 0.2s;
}
.info-card:hover { border-left-color: #FFA500; }
.info-card h4 { color: #00BFFF; margin: 0 0 6px 0; font-size: 15px; }
.info-card p  { color: #cdd6f4; margin: 0; font-size: 13px; line-height: 1.5; }
.info-card .normal { color: #00e676; font-weight: bold; }
.info-card .warn   { color: #FFA500; font-weight: bold; }
.info-card .danger { color: #FF4B4B; font-weight: bold; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 10px;
    border: 1px solid #2d4a7a;
}
.insight-box p { color: #b0c4de; font-size: 13px; margin: 0; }

/* Filter panel */
.filter-panel {
    background: #0f1f3d;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid #1e3a5f;
}

/* Risk level badges */
.badge-low    { background: #1b4332; color: #00e676; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-medium { background: #7b4f12; color: #FFA500; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-high   { background: #6b1a1a; color: #FF4B4B; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e4a6e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("stroke_dataset_cleaned_final.csv")
    df["stroke_label"] = df["stroke"].apply(lambda x: "Stroke" if x == 1 else "Tidak Stroke")
    bins = [0, 18, 35, 50, 65, 100]
    labels_age_bins = ['0-18', '19-35', '36-50', '51-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels_age_bins, right=False)
    df["health_risk_score"] = df["hypertension"] + df["heart_disease"]
    # BMI category
    def bmi_cat(b):
        if b < 18.5: return "Kurus"
        elif b < 25: return "Normal"
        elif b < 30: return "Overweight"
        else: return "Obesitas"
    df['bmi_category'] = df['bmi'].apply(bmi_cat)
    # Glucose category
    def gluc_cat(g):
        if g < 100: return "Normal"
        elif g < 126: return "Pradiabetes"
        else: return "Diabetes"
    df['glucose_category'] = df['avg_glucose_level'].apply(gluc_cat)
    return df

df = load_data()

stroke_pct = df['stroke'].mean() * 100
STROKE_COLOR_MAP = {"Stroke": "#FF4B4B", "Tidak Stroke": "#00BFFF"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    font_family="Inter",
    hoverlabel=dict(bgcolor="#1a2744", font_color="white", font_size=13),
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=70)
st.sidebar.title("🧠 Stroke Dashboard")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "🤖 Prediksi Stroke",
        "📊 Overview",
        "🔍 EDA & Distribusi",
        "⚠️ Faktor Risiko",
        "🧪 A/B Testing & Evaluasi",
        "📋 Kesimpulan"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background:#0f1f3d;border-radius:10px;padding:12px 16px;">
  <div style="color:#90caf9;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Dataset Info</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="color:#cdd6f4;font-size:13px;">Total Pasien</span>
    <span style="color:white;font-weight:700;font-size:13px;">{len(df):,}</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="color:#cdd6f4;font-size:13px;">Kasus Stroke</span>
    <span style="color:#FF4B4B;font-weight:700;font-size:13px;">{df['stroke'].sum():,}</span>
  </div>
  <div style="display:flex;justify-content:space-between;">
    <span style="color:#cdd6f4;font-size:13px;">Prevalensi</span>
    <span style="color:#FFA500;font-weight:700;font-size:13px;">{stroke_pct:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: PREDIKSI STROKE — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🤖 Prediksi Stroke":
    st.markdown("""
    <div class="main-header">
      <h1>🤖 Prediksi Risiko Stroke</h1>
      <p>Masukkan data pasien untuk menghitung estimasi risiko stroke berdasarkan faktor klinis</p>
    </div>
    """, unsafe_allow_html=True)

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.markdown("#### 🩺 Data Klinis")
        age = st.slider("Usia (tahun)", 0, 100, 45,
                        help="Risiko stroke meningkat signifikan di atas 50 tahun")
        hypertension = st.radio("Hipertensi", [0, 1],
                                format_func=lambda x: "✅ Tidak" if x == 0 else "⚠️ Ya",
                                horizontal=True)
        heart_disease = st.radio("Penyakit Jantung", [0, 1],
                                 format_func=lambda x: "✅ Tidak" if x == 0 else "⚠️ Ya",
                                 horizontal=True)
        glucose = st.slider("Kadar Glukosa (mg/dL)", 50, 300, 100,
                            help="Normal: 70–100 mg/dL (puasa)")

        # Glucose indicator
        if glucose < 100:
            st.success(f"🟢 **Normal** ({glucose} mg/dL) — Kadar gula darah sehat")
        elif glucose < 126:
            st.warning(f"🟡 **Pradiabetes** ({glucose} mg/dL) — Waspadai risiko diabetes")
        else:
            st.error(f"🔴 **Diabetes** ({glucose} mg/dL) — Kadar gula tinggi, risiko meningkat")

    with col_input2:
        st.markdown("#### 📏 Hitung BMI Otomatis")
        berat = st.slider("Berat Badan (kg)", 20, 200, 65)
        tinggi = st.slider("Tinggi Badan (cm)", 100, 250, 165)
        bmi = berat / (tinggi / 100) ** 2

        # BMI Gauge
        fig_bmi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bmi,
            number={"suffix": " kg/m²", "font": {"size": 24}},
            title={"text": "BMI", "font": {"size": 14}},
            gauge={
                "axis": {"range": [10, 45], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#FF4B4B" if bmi >= 30 else ("#FFA500" if bmi >= 25 else ("#00e676" if bmi >= 18.5 else "#90caf9"))},
                "steps": [
                    {"range": [10, 18.5], "color": "#1a3a5c"},
                    {"range": [18.5, 25], "color": "#1b4332"},
                    {"range": [25, 30], "color": "#7b4f12"},
                    {"range": [30, 45], "color": "#6b1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": bmi}
            }
        ))
        fig_bmi.update_layout(height=200, **PLOTLY_LAYOUT, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_bmi, use_container_width=True)

        bmi_labels = {(10,18.5): ("🔵 Underweight","#90caf9"),
                      (18.5,25): ("🟢 Normal","#00e676"),
                      (25,30): ("🟡 Overweight","#FFA500"),
                      (30,99): ("🔴 Obesitas","#FF4B4B")}
        for (lo,hi),(lbl,clr) in bmi_labels.items():
            if lo <= bmi < hi:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:{clr};font-size:16px'>{lbl} — BMI {bmi:.1f}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Predict button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_btn = st.button("🔍 Analisis Risiko Sekarang", use_container_width=True, type="primary")

    if predict_btn:
        risk_score = (age * 0.03 + glucose * 0.01 + bmi * 0.01
                      + hypertension * 2 + heart_disease * 2)
        probability = min(risk_score / 10, 1)

        # Result banner
        if probability > 0.65:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#6b1a1a,#3d0a0a);border-radius:12px;
                padding:20px 24px;border:1px solid #FF4B4B;text-align:center;margin-bottom:16px;">
                <div style="font-size:22px;font-weight:700;color:#FF4B4B">⛔ RISIKO TINGGI — {probability*100:.1f}%</div>
                <div style="color:#ffaaaa;margin-top:6px;">Segera konsultasi ke dokter spesialis saraf atau neurologi</div>
            </div>""", unsafe_allow_html=True)
        elif probability > 0.35:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#7b4f12,#3d2808);border-radius:12px;
                padding:20px 24px;border:1px solid #FFA500;text-align:center;margin-bottom:16px;">
                <div style="font-size:22px;font-weight:700;color:#FFA500">⚠️ RISIKO SEDANG — {probability*100:.1f}%</div>
                <div style="color:#ffd080;margin-top:6px;">Perlu perhatian khusus dan pemantauan rutin</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#1b4332,#0a2119);border-radius:12px;
                padding:20px 24px;border:1px solid #00e676;text-align:center;margin-bottom:16px;">
                <div style="font-size:22px;font-weight:700;color:#00e676">✅ RISIKO RENDAH — {probability*100:.1f}%</div>
                <div style="color:#80ffb8;margin-top:6px;">Tetap jaga pola hidup sehat dan rutin cek kesehatan</div>
            </div>""", unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            # Main gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Skor Risiko Stroke", "font": {"size": 16}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                    "bar": {"color": "#FF4B4B" if probability > 0.5 else "#00C853", "thickness": 0.3},
                    "steps": [
                        {"range": [0, 35], "color": "#1b4332"},
                        {"range": [35, 65], "color": "#7b4f12"},
                        {"range": [65, 100], "color": "#6b1a1a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=320, **PLOTLY_LAYOUT, margin=dict(t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r2:
            # Factor contribution — horizontal bar with colors
            factors_data = {
                "Faktor": ["Penyakit Jantung", "Hipertensi", "Usia", "BMI", "Glukosa"],
                "Kontribusi": [heart_disease*2, hypertension*2, age*0.03, bmi*0.01, glucose*0.01],
                "Kategori": ["Klinis", "Klinis", "Demografis", "Fisik", "Metabolik"]
            }
            factors = pd.DataFrame(factors_data).sort_values("Kontribusi")
            max_val = factors["Kontribusi"].max()
            factors["Persentase"] = (factors["Kontribusi"] / max_val * 100).round(1) if max_val > 0 else 0

            fig_bar = px.bar(factors, x="Kontribusi", y="Faktor", orientation='h',
                             color="Kontribusi",
                             color_continuous_scale=["#1b4332","#FFA500","#FF4B4B"],
                             text=factors["Kontribusi"].round(2),
                             title="📊 Kontribusi Tiap Faktor Risiko")
            fig_bar.update_traces(textposition='outside', textfont_size=12)
            fig_bar.update_layout(height=320, **PLOTLY_LAYOUT,
                                  coloraxis_showscale=False,
                                  margin=dict(t=50, b=20, l=10, r=60))
            st.plotly_chart(fig_bar, use_container_width=True)






# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Overview":
    st.markdown("""
    <div class="main-header">
      <h1>📊 Overview Dataset Stroke Prediction</h1>
      <p>Dataset: Stroke Prediction Dataset (Kaggle - fedesoriano) | Sudah melalui Data Wrangling</p>
    </div>
    """, unsafe_allow_html=True)

    n_stroke   = int(df['stroke'].sum())
    n_nonstroke = len(df) - n_stroke

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card kpi-blue">
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-label">Total Pasien</div>
            <div class="kpi-delta" style="color:#90caf9">📁 11 Fitur</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card kpi-red">
            <div class="kpi-value">{n_stroke:,}</div>
            <div class="kpi-label">Kasus Stroke</div>
            <div class="kpi-delta" style="color:#FF4B4B">{stroke_pct:.1f}% dari total</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card kpi-orange">
            <div class="kpi-value">{df['age'].mean():.0f}</div>
            <div class="kpi-label">Rata-rata Usia</div>
            <div class="kpi-delta" style="color:#FFA500">tahun</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card kpi-green">
            <div class="kpi-value">{df['bmi'].mean():.1f}</div>
            <div class="kpi-label">Rata-rata BMI</div>
            <div class="kpi-delta" style="color:#00e676">kg/m²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("🍩 Distribusi Kasus Stroke")
        stroke_counts = df['stroke_label'].value_counts().reset_index()
        stroke_counts.columns = ['Status', 'Jumlah']
        fig = px.pie(stroke_counts, values='Jumlah', names='Status',
                     color='Status', color_discrete_map=STROKE_COLOR_MAP, hole=0.55)
        fig.update_traces(
            textinfo='percent+label', textfont_size=14,
            pull=[0.04, 0],
            hovertemplate="<b>%{label}</b><br>%{value:,} pasien<br>%{percent}<extra></extra>"
        )
        fig.update_layout(height=320, **PLOTLY_LAYOUT,
                          annotations=[dict(text=f"<b>{len(df):,}</b><br>Pasien",
                                            x=0.5, y=0.5, font_size=16, font_color="white",
                                            showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📋 Tentang Dataset Ini")
        pct = n_stroke / len(df) * 100
        st.markdown(f"""
        <div style="background:#1a2744;border-left:4px solid #00BFFF;border-radius:8px;
                    padding:14px 18px;margin-bottom:10px;">
          <div style="color:#00BFFF;font-weight:bold;font-size:14px;margin-bottom:8px;">🏥 Apa isi dataset ini?</div>
          <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
            Dataset ini berisi catatan kesehatan <b style="color:white">{len(df):,} pasien</b>
            yang masing-masing diukur 11 indikator kesehatan seperti usia, tekanan darah, kadar gula darah, dan gaya hidup.<br><br>
            Dari seluruh pasien, sebanyak <b style="color:#FF4B4B">{n_stroke:,} orang ({pct:.1f}%)</b> tercatat pernah mengalami stroke.<br><br>
            Tujuan: <b style="color:#00e676">Memahami faktor risiko stroke & membangun model prediksi dini.</b>
          </p>
        </div>
        <div style="background:#1a2744;border-left:4px solid #FFA500;border-radius:8px;
                    padding:14px 18px;">
          <div style="color:#FFA500;font-weight:bold;font-size:14px;margin-bottom:8px;">📌 Faktor yang diteliti</div>
          <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
            <b style="color:white">Kondisi medis:</b> usia, hipertensi, penyakit jantung, kadar gula, BMI<br>
            <b style="color:white">Gaya hidup:</b> status merokok, jenis pekerjaan<br>
            <b style="color:white">Demografi:</b> jenis kelamin, status menikah, tempat tinggal
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Statistik interaktif
    st.subheader("📈 Statistik Variabel Numerik — Interaktif")

    _num_label_map = {"age":"Usia (tahun)","avg_glucose_level":"Glukosa (mg/dL)","bmi":"BMI"}
    stat_var = st.selectbox("Pilih variabel untuk dilihat distribusinya:",
                             ["age", "avg_glucose_level", "bmi"],
                             format_func=lambda x: _num_label_map[x])

    fig_hist = px.histogram(df, x=stat_var, color='stroke_label', barmode='overlay',
                             nbins=40, opacity=0.75,
                             color_discrete_map=STROKE_COLOR_MAP,
                             category_orders={"stroke_label":["Tidak Stroke","Stroke"]},
                             labels={stat_var: _num_label_map[stat_var], 'stroke_label':'Status'},
                             title=f"Distribusi {_num_label_map[stat_var]} per Status Stroke")
    fig_hist.update_layout(height=320, **PLOTLY_LAYOUT, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Data dictionary
    st.subheader("📖 Panduan Variabel")
    dict_cards = [
        ("🧬", "gender", "Jenis Kelamin", "Kategorikal", "Male / Female",
         "Jenis kelamin biologis pasien.", None),
        ("🎂", "age", "Usia", "Numerik", "0 – 82 tahun",
         "Usia pasien. Risiko stroke meningkat signifikan di atas 50 tahun.",
         "Risiko tinggi: <span class='danger'>≥ 50 tahun</span>"),
        ("💊", "hypertension", "Hipertensi", "Biner (0/1)", "0=Tidak, 1=Ya",
         "Tekanan darah tinggi (≥ 140/90 mmHg). Salah satu faktor risiko stroke terbesar.",
         "Normal: <span class='normal'>< 120/80</span> | Tinggi: <span class='danger'>≥ 140/90 mmHg</span>"),
        ("❤️", "heart_disease", "Penyakit Jantung", "Biner (0/1)", "0=Tidak, 1=Ya",
         "Riwayat penyakit jantung. Meningkatkan risiko gumpalan darah ke otak.", None),
        ("🍬", "avg_glucose_level", "Kadar Glukosa", "Numerik", "55 – 272 mg/dL",
         "Rata-rata kadar gula darah. Glukosa tinggi meningkatkan risiko stroke.",
         "Normal: <span class='normal'>70–100</span> | Pradiabetes: <span class='warn'>100–125</span> | Diabetes: <span class='danger'>≥ 126</span>"),
        ("⚖️", "bmi", "BMI", "Numerik", "10.3 – 97.6",
         "Indeks Massa Tubuh = Berat (kg) ÷ Tinggi² (m²).",
         "Kurus: <span class='warn'>< 18.5</span> | Normal: <span class='normal'>18.5–24.9</span> | Overweight: <span class='warn'>25–29.9</span> | Obesitas: <span class='danger'>≥ 30</span>"),
        ("🚬", "smoking_status", "Status Merokok", "Kategorikal", "formerly smoked / never / smokes / Unknown",
         "Merokok merusak pembuluh darah dan meningkatkan risiko stroke iskemik.", None),
        ("🎯", "stroke", "Stroke (Target)", "Biner (0/1)", "0=Tidak, 1=Ya",
         "Variabel target — apakah pasien pernah stroke.", None),
    ]
    for icon, col_name, label, dtype, values, desc, ref in dict_cards:
        ref_html = f"<br><small>{ref}</small>" if ref else ""
        st.markdown(f"""
        <div class="info-card">
          <h4>{icon} {label} <code style="font-size:11px;color:#90caf9">{col_name}</code>
            &nbsp;<span style="font-size:11px;background:#0f3460;border-radius:10px;padding:2px 8px;color:#64b5f6">{dtype}</span>
          </h4>
          <p><b style="color:#90caf9">Nilai:</b> {values}</p>
          <p>{desc}{ref_html}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA & DISTRIBUSI — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 EDA & Distribusi":
    st.markdown("""
    <div class="main-header">
      <h1>🔍 Exploratory Data Analysis</h1>
      <p>Jelajahi distribusi data dan temukan pola tersembunyi di balik faktor risiko stroke</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Variabel Numerik", "🏷️ Variabel Kategorikal", "🔥 Korelasi", "🌐 Scatter 3D"])

    with tab1:
        st.subheader("Distribusi Variabel Numerik")

        # Filter panel
        c_f1, c_f2 = st.columns([1,2])
        with c_f1:
            filter_num = st.radio("Filter Status:", ["Semua","Stroke","Tidak Stroke"],
                                   horizontal=True, key="fn")
        with c_f2:
            age_range = st.slider("Filter Usia:", 0, 100, (0, 100), key="age_range_eda")

        df_n = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
        if filter_num != "Semua":
            df_n = df_n[df_n["stroke_label"] == filter_num]

        st.markdown(f"<div style='color:#90caf9;font-size:13px;margin-bottom:8px'>Menampilkan <b style='color:white'>{len(df_n):,}</b> pasien</div>", unsafe_allow_html=True)

        num_cols   = ['age', 'avg_glucose_level', 'bmi']
        num_labels = {'age':'Usia (tahun)','avg_glucose_level':'Glukosa (mg/dL)','bmi':'BMI'}

        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(num_cols):
            fig = px.histogram(
                df_n, x=col,
                color='stroke_label' if filter_num == "Semua" else None,
                barmode='overlay', nbins=35, opacity=0.8,
                color_discrete_map=STROKE_COLOR_MAP,
                category_orders={"stroke_label":["Tidak Stroke","Stroke"]},
                labels={col: num_labels[col], 'stroke_label':'Status'},
                title=f"Distribusi {num_labels[col]}"
            )
            fig.update_layout(height=280, **PLOTLY_LAYOUT, bargap=0.05,
                              margin=dict(t=40,b=30,l=30,r=10), showlegend=(i==0))
            [c1,c2,c3][i].plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Perbandingan Rata-rata per Status Stroke")
        viol_var = st.selectbox("Variabel:", num_cols,
                                 format_func=lambda x: num_labels[x], key="viol_var")

        # Hitung mean & std per grup
        grp_stats = df.groupby('stroke_label')[viol_var].agg(['mean','std','count']).reset_index()
        grp_stats.columns = ['Status','Mean','Std','Count']
        grp_stats['SE'] = grp_stats['Std'] / grp_stats['Count']**0.5  # standard error

        fig_mean = go.Figure()
        colors = {"Stroke": "#FF4B4B", "Tidak Stroke": "#00BFFF"}
        for _, row in grp_stats.iterrows():
            fig_mean.add_trace(go.Bar(
                x=[row['Status']],
                y=[row['Mean']],
                name=row['Status'],
                marker_color=colors[row['Status']],
                error_y=dict(type='data', array=[row['SE']*1.96], color='white', thickness=2, width=8),
                text=[f"<b>{row['Mean']:.1f}</b>"],
                textposition='outside',
                textfont=dict(size=14, color='white'),
                width=0.4
            ))

        # Tambah nilai min, median, max sebagai annotation
        for status in ['Stroke','Tidak Stroke']:
            sub = df[df['stroke_label']==status][viol_var]
            fig_mean.add_annotation(
                x=status, y=grp_stats[grp_stats['Status']==status]['Mean'].values[0],
                text=f"  Median: {sub.median():.1f} | Min: {sub.min():.1f} | Max: {sub.max():.1f}",
                showarrow=False, xanchor='left',
                font=dict(size=11, color='#90caf9'),
                yshift=-28
            )

        fig_mean.update_layout(
            title=f"Rata-rata {num_labels[viol_var]} — Stroke vs Tidak Stroke",
            yaxis_title=num_labels[viol_var],
            height=400, **PLOTLY_LAYOUT,
            showlegend=False,
            bargap=0.4,
            yaxis=dict(zeroline=False)
        )
        st.plotly_chart(fig_mean, use_container_width=True)

        # Insight otomatis
        mean_stroke    = grp_stats[grp_stats['Status']=='Stroke']['Mean'].values[0]
        mean_nonstroke = grp_stats[grp_stats['Status']=='Tidak Stroke']['Mean'].values[0]
        diff_pct = abs(mean_stroke - mean_nonstroke) / mean_nonstroke * 100
        direction = "lebih tinggi" if mean_stroke > mean_nonstroke else "lebih rendah"
        st.markdown(f"""<div class="insight-box">
        <p>📌 Rata-rata <b>{num_labels[viol_var]}</b> pada pasien <b style="color:#FF4B4B">Stroke</b>
        adalah <b>{mean_stroke:.1f}</b>, sedangkan <b style="color:#00BFFF">Tidak Stroke</b> rata-rata
        <b>{mean_nonstroke:.1f}</b>. Pasien stroke <b>{direction} {diff_pct:.1f}%</b> dibanding non-stroke.
        Error bar menunjukkan interval kepercayaan 95%.</p></div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader("Distribusi Variabel Kategorikal")
        chart_type = st.radio("Tipe Chart:", ["Bar Chart", "Treemap"], horizontal=True)
        filter_cat = st.radio("Filter Status:", ["Semua","Stroke","Tidak Stroke"],
                               horizontal=True, key="fc")
        df_c = df if filter_cat == "Semua" else df[df["stroke_label"]==filter_cat]

        cat_cols   = ['gender','ever_married','work_type','Residence_type','smoking_status']
        cat_labels = {'gender':'Jenis Kelamin','ever_married':'Status Menikah',
                      'work_type':'Jenis Pekerjaan','Residence_type':'Tempat Tinggal',
                      'smoking_status':'Status Merokok'}

        sel_cat = st.selectbox("Pilih Variabel:", cat_cols,
                                format_func=lambda x: cat_labels[x], key="cat_sel")

        if chart_type == "Bar Chart":
            if filter_cat == "Semua":
                grp = df_c.groupby([sel_cat,'stroke_label']).size().reset_index(name='count')
                # Hitung stroke rate per kategori
                total_per_cat = df_c.groupby(sel_cat).size().reset_index(name='total')
                stroke_per_cat = df_c[df_c['stroke']==1].groupby(sel_cat).size().reset_index(name='n_stroke')
                rate_df = total_per_cat.merge(stroke_per_cat, on=sel_cat, how='left').fillna(0)
                rate_df['rate'] = (rate_df['n_stroke'] / rate_df['total'] * 100).round(1)

                fig = px.bar(grp, x=sel_cat, y='count', color='stroke_label', barmode='group',
                             color_discrete_map=STROKE_COLOR_MAP,
                             labels={'count':'Jumlah', sel_cat: cat_labels[sel_cat], 'stroke_label':'Status'},
                             title=f"Distribusi {cat_labels[sel_cat]}", text_auto=True)
                fig.update_traces(textposition='outside')
                fig.update_layout(height=360, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                # Stroke rate table
                st.markdown("**📊 Stroke Rate per Kategori:**")
                rate_df_show = rate_df.rename(columns={sel_cat: cat_labels[sel_cat],
                                                        'total':'Total Pasien',
                                                        'n_stroke':'Kasus Stroke',
                                                        'rate':'Stroke Rate (%)'})
                st.dataframe(rate_df_show.set_index(cat_labels[sel_cat]), use_container_width=True)
            else:
                grp = df_c.groupby(sel_cat).size().reset_index(name='count')
                clr = "#FF4B4B" if filter_cat == "Stroke" else "#00BFFF"
                fig = px.bar(grp, x=sel_cat, y='count', color_discrete_sequence=[clr],
                             labels={'count':'Jumlah', sel_cat: cat_labels[sel_cat]},
                             title=f"Distribusi {cat_labels[sel_cat]} — {filter_cat}", text_auto=True)
                fig.update_traces(textposition='outside')
                fig.update_layout(height=360, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Treemap
            grp = df.groupby([sel_cat,'stroke_label']).size().reset_index(name='count')
            fig_tree = px.treemap(grp, path=[sel_cat, 'stroke_label'], values='count',
                                   color='stroke_label', color_discrete_map=STROKE_COLOR_MAP,
                                   title=f"Treemap — {cat_labels[sel_cat]} × Status Stroke")
            fig_tree.update_layout(height=420, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_tree, use_container_width=True)

    with tab3:
        st.subheader("🔥 Heatmap Korelasi Interaktif")
        vars_sel = st.multiselect("Pilih variabel:",
                                   ['age','avg_glucose_level','bmi','hypertension','heart_disease','stroke'],
                                   default=['age','avg_glucose_level','bmi','hypertension','heart_disease','stroke'])
        if len(vars_sel) >= 2:
            corr = df[vars_sel].corr()
            label_map = {'age':'Usia','avg_glucose_level':'Glukosa','bmi':'BMI',
                         'hypertension':'Hipertensi','heart_disease':'P.Jantung','stroke':'Stroke'}
            corr.index   = [label_map.get(v,v) for v in corr.index]
            corr.columns = [label_map.get(v,v) for v in corr.columns]

            fig_heat = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=13, color="white"),
                hoverongaps=False,
                hovertemplate="<b>%{y} × %{x}</b><br>Korelasi: %{z:.3f}<extra></extra>"
            ))
            fig_heat.update_layout(height=420, **PLOTLY_LAYOUT, margin=dict(t=40,b=30,l=80,r=20))
            st.plotly_chart(fig_heat, use_container_width=True)

            # Rank korelasi terhadap stroke
            if 'stroke' in vars_sel:
                st.markdown("#### 📊 Ranking Korelasi terhadap Stroke")
                corr_stroke = df[vars_sel].corr()['stroke'].drop('stroke').abs().sort_values(ascending=False)
                def interpret(v):
                    if v >= 0.4: return "🔴 Kuat"
                    elif v >= 0.2: return "🟠 Sedang"
                    elif v >= 0.1: return "🟡 Lemah"
                    else: return "⚪ Sangat Lemah"
                rows = [{"Variabel": label_map.get(v,v),
                         "Korelasi": round(val,3),
                         "Kekuatan": interpret(val)}
                        for v, val in corr_stroke.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("🌐 Scatter 3D — Usia × Glukosa × BMI")
        st.markdown("Putar grafik untuk melihat pola dari berbagai sudut pandang.")

        sample_size = st.slider("Jumlah sampel:", 500, len(df), min(2000, len(df)), 100, key="scatter3d_n")
        df_samp = df.sample(n=sample_size, random_state=42)

        fig_3d = px.scatter_3d(df_samp, x='age', y='avg_glucose_level', z='bmi',
                                color='stroke_label', color_discrete_map=STROKE_COLOR_MAP,
                                opacity=0.65, size_max=6,
                                labels={'age':'Usia','avg_glucose_level':'Glukosa','bmi':'BMI','stroke_label':'Status'},
                                title="Scatter 3D: Usia × Glukosa × BMI")
        fig_3d.update_layout(height=550, **PLOTLY_LAYOUT,
                              scene=dict(
                                  xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", color="white"),
                                  yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", color="white"),
                                  zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e3a5f", color="white"),
                              ))
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown("""<div class="insight-box">
        <p>📌 Pasien stroke (merah) cenderung terkonsentrasi di area <b>usia tinggi + glukosa tinggi</b>.
        Coba putar 3D plot untuk melihat pola distribusinya secara lebih jelas.</p></div>""",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FAKTOR RISIKO — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "⚠️ Faktor Risiko":
    st.markdown("""
    <div class="main-header">
      <h1>⚠️ Analisis Faktor Risiko Stroke</h1>
      <p>Setiap faktor disertai visualisasi interaktif dan penjelasan klinis</p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive filter
    with st.expander("🎛️ Filter Data Global", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            age_filter = st.slider("Rentang Usia:", 0, 100, (0,100), key="risk_age")
        with fc2:
            gluc_filter = st.slider("Rentang Glukosa:", 50, 300, (50,300), key="risk_gluc")
        with fc3:
            gender_filter = st.multiselect("Jenis Kelamin:", df['gender'].unique().tolist(),
                                            default=df['gender'].unique().tolist(), key="risk_gender")
    df_r = df[(df['age'].between(*age_filter)) &
              (df['avg_glucose_level'].between(*gluc_filter)) &
              (df['gender'].isin(gender_filter))]
    st.markdown(f"<div style='color:#90caf9;font-size:13px;margin-bottom:12px'>Filter aktif: <b style='color:white'>{len(df_r):,}</b> pasien</div>", unsafe_allow_html=True)

    # Tabs per faktor
    t1, t2, t3, t4, t5 = st.tabs(["🎂 Usia", "🍬 Glukosa", "💊 Hipertensi & Jantung", "⚖️ BMI", "🔵 Scatter Gabungan"])

    with t1:
        st.subheader("🎂 Usia — Faktor Risiko Terkuat")
        age_stroke = df_r.groupby(['age_group','stroke_label']).size().reset_index(name='count')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(age_stroke, x='age_group', y='count', color='stroke_label', barmode='group',
                         color_discrete_map=STROKE_COLOR_MAP,
                         labels={'count':'Jumlah','age_group':'Kel. Usia','stroke_label':'Status'},
                         title="Distribusi Stroke per Kelompok Usia", text_auto=True)
            fig.update_traces(textposition='outside')
            fig.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Stroke rate per age group
            age_total = df_r.groupby('age_group').size().reset_index(name='total')
            age_stroke_n = df_r[df_r['stroke']==1].groupby('age_group').size().reset_index(name='n_stroke')
            age_rate = age_total.merge(age_stroke_n, on='age_group', how='left').fillna(0)
            age_rate['rate'] = (age_rate['n_stroke'] / age_rate['total'] * 100).round(1)
            fig2 = px.line(age_rate, x='age_group', y='rate', markers=True,
                            title="Stroke Rate (%) per Kelompok Usia",
                            labels={'age_group':'Kelompok Usia','rate':'Stroke Rate (%)'},
                            color_discrete_sequence=["#FF4B4B"])
            fig2.update_traces(line_width=3, marker_size=10)
            fig2.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <p>📌 <b>Mengapa usia berpengaruh?</b> Pembuluh darah semakin kaku dan rentan menyempit (aterosklerosis)
        seiring bertambahnya usia. Pasien stroke rata-rata berusia <b>~68 tahun</b> vs ~41 tahun bagi yang tidak stroke.
        Risiko meningkat tajam pada kelompok <b>51–65 tahun</b> dan <b>65+ tahun</b>.</p></div>""", unsafe_allow_html=True)

    with t2:
        st.subheader("🍬 Kadar Glukosa — Indikator Diabetes")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_r, x='avg_glucose_level', color='stroke_label', barmode='overlay',
                                nbins=35, opacity=0.75, color_discrete_map=STROKE_COLOR_MAP,
                                category_orders={"stroke_label":["Tidak Stroke","Stroke"]},
                                labels={'avg_glucose_level':'Glukosa (mg/dL)','stroke_label':'Status'},
                                title="Distribusi Glukosa berdasarkan Status Stroke")
            fig.add_vline(x=126, line_dash="dash", line_color="#FFA500",
                          annotation_text="Batas Diabetes (126)", annotation_font_color="#FFA500")
            fig.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gluc_stroke = df_r.groupby(['glucose_category','stroke_label']).size().reset_index(name='count')
            fig2 = px.bar(gluc_stroke, x='glucose_category', y='count', color='stroke_label', barmode='group',
                           color_discrete_map=STROKE_COLOR_MAP,
                           category_orders={"glucose_category":["Normal","Pradiabetes","Diabetes"]},
                           labels={'count':'Jumlah','glucose_category':'Kategori Glukosa','stroke_label':'Status'},
                           title="Stroke per Kategori Glukosa", text_auto=True)
            fig2.update_traces(textposition='outside')
            fig2.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <p>📌 <b>Mengapa glukosa berpengaruh?</b> Gula darah tinggi merusak dinding pembuluh darah dan
        membuat darah lebih mudah membeku. Pasien stroke rata-rata glukosa <b>~133 mg/dL</b> vs ~103 mg/dL.
        Garis oranye menunjukkan batas diabetes (<b>≥ 126 mg/dL</b>).</p></div>""", unsafe_allow_html=True)

    with t3:
        st.subheader("💊 Hipertensi & Penyakit Jantung")
        hrs = df_r.groupby(['health_risk_score','stroke_label']).size().reset_index(name='count')
        hrs['health_risk_score'] = hrs['health_risk_score'].map(
            {0:'Tidak Ada (0)', 1:'Salah Satu (1)', 2:'Keduanya (2)'})
        fig_hrs = px.bar(hrs, x='health_risk_score', y='count', color='stroke_label', barmode='group',
                          color_discrete_map=STROKE_COLOR_MAP,
                          labels={'health_risk_score':'Health Risk Score','count':'Jumlah','stroke_label':'Status'},
                          title="Health Risk Score (Hipertensi + Penyakit Jantung) vs Stroke", text_auto=True)
        fig_hrs.update_traces(textposition='outside')
        fig_hrs.update_layout(height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_hrs, use_container_width=True)

        c1, c2 = st.columns(2)
        for col, title, cname in [('hypertension','Hipertensi vs Stroke','hyp'),
                                    ('heart_disease','Penyakit Jantung vs Stroke','hd')]:
            grp = df_r.groupby([col,'stroke_label']).size().reset_index(name='count')
            grp[col] = grp[col].map({0:'Tidak',1:'Ya'})
            fig = px.bar(grp, x=col, y='count', color='stroke_label', barmode='group',
                          color_discrete_map=STROKE_COLOR_MAP, title=title,
                          labels={col: col.replace('_',' ').title(),'count':'Jumlah','stroke_label':'Status'},
                          text_auto=True)
            fig.update_traces(textposition='outside')
            fig.update_layout(height=300, **PLOTLY_LAYOUT)
            (c1 if col=='hypertension' else c2).plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <p>📌 Hipertensi memberikan tekanan berlebih pada pembuluh darah otak, sedangkan penyakit jantung
        meningkatkan risiko gumpalan yang menyumbat arteri otak. Proporsi hipertensi pada pasien stroke
        <b>26.5%</b> vs <b>9.5%</b> pada yang tidak stroke.</p></div>""", unsafe_allow_html=True)

    with t4:
        st.subheader("⚖️ BMI — Peran Tidak Langsung")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_r, x='bmi', color='stroke_label', barmode='overlay',
                                nbins=35, opacity=0.75, color_discrete_map=STROKE_COLOR_MAP,
                                category_orders={"stroke_label":["Tidak Stroke","Stroke"]},
                                title="Distribusi BMI per Status Stroke",
                                labels={'bmi':'BMI','stroke_label':'Status'})
            for val, lbl, clr in [(18.5,"Kurus|Normal","#90caf9"),(25,"Normal|Overweight","#FFA500"),(30,"Overweight|Obesitas","#FF4B4B")]:
                fig.add_vline(x=val, line_dash="dash", line_color=clr,
                              annotation_text=lbl, annotation_font_color=clr, annotation_font_size=11)
            fig.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bmi_stroke = df_r.groupby(['bmi_category','stroke_label']).size().reset_index(name='count')
            fig2 = px.bar(bmi_stroke, x='bmi_category', y='count', color='stroke_label', barmode='group',
                           color_discrete_map=STROKE_COLOR_MAP,
                           category_orders={"bmi_category":["Kurus","Normal","Overweight","Obesitas"]},
                           title="Stroke per Kategori BMI", text_auto=True,
                           labels={'bmi_category':'Kategori BMI','count':'Jumlah','stroke_label':'Status'})
            fig2.update_traces(textposition='outside')
            fig2.update_layout(height=340, **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with t5:
        st.subheader("🔵 Scatter Interaktif — Kombinasi Faktor")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_var = st.selectbox("Sumbu X:", ['age','avg_glucose_level','bmi'],
                                  format_func=lambda x: {'age':'Usia','avg_glucose_level':'Glukosa','bmi':'BMI'}[x], key="scx")
        with sc2:
            y_var = st.selectbox("Sumbu Y:", ['avg_glucose_level','bmi','age'],
                                  format_func=lambda x: {'age':'Usia','avg_glucose_level':'Glukosa','bmi':'BMI'}[x], key="scy")
        with sc3:
            size_var = st.selectbox("Ukuran titik:", ['None','age','avg_glucose_level','bmi'],
                                     format_func=lambda x: {'None':'Sama','age':'Usia','avg_glucose_level':'Glukosa','bmi':'BMI'}[x], key="scsize")

        df_sc = df_r.sample(min(2000, len(df_r)), random_state=1)
        fig_sc = px.scatter(df_sc, x=x_var, y=y_var, color='stroke_label',
                             size=size_var if size_var != 'None' else None,
                             size_max=15, opacity=0.6,
                             color_discrete_map=STROKE_COLOR_MAP,
                             labels={'stroke_label':'Status'},
                             title=f"Scatter: {x_var} vs {y_var}")
        fig_sc.update_layout(height=450, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <p>📌 Coba pilih kombinasi <b>Usia (X) × Glukosa (Y)</b> — pasien stroke (merah) terkonsentrasi
        di pojok kanan atas. Garis tren (lowess) membantu melihat pola nonlinear.</p></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: A/B TESTING & EVALUASI — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🧪 A/B Testing & Evaluasi":
    st.markdown("""
    <div class="main-header">
      <h1>🧪 A/B Testing & Evaluasi Model</h1>
      <p>Uji signifikansi statistik faktor risiko dan evaluasi performa prediksi</p>
    </div>
    """, unsafe_allow_html=True)

    tab_ab, tab_cm = st.tabs(["📐 Uji Chi-Square", "🔲 Confusion Matrix"])

    with tab_ab:
        st.subheader("A/B Testing — Uji Statistik Faktor Risiko")
        st.markdown("""
        <div class="insight-box" style="margin-bottom:16px">
        <p><b>A/B Testing</b> di sini menjawab: <i>"Apakah faktor risiko tertentu secara statistik
        berhubungan dengan stroke?"</i> Menggunakan uji Chi-Square untuk variabel kategorikal.</p></div>
        """, unsafe_allow_html=True)

        col_ab1, col_ab2 = st.columns([2,1])
        with col_ab1:
            ab_var = st.selectbox("Pilih variabel:", ["hypertension","heart_disease","ever_married","gender"],
                                   format_func=lambda x: {"hypertension":"Hipertensi","heart_disease":"Penyakit Jantung",
                                                           "ever_married":"Status Menikah","gender":"Jenis Kelamin"}[x])
        with col_ab2:
            alpha = st.select_slider("Level signifikansi (α):", options=[0.01, 0.05, 0.10], value=0.05)

        contingency = pd.crosstab(df[ab_var], df['stroke'])
        chi2, p, dof, expected = chi2_contingency(contingency)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""<div class="kpi-card kpi-blue" style="padding:16px">
                <div class="kpi-value" style="font-size:24px">{chi2:.3f}</div>
                <div class="kpi-label">Chi-Square</div></div>""", unsafe_allow_html=True)
        with k2:
            p_display = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
            p_color = "kpi-red" if p < alpha else "kpi-green"
            st.markdown(f"""<div class="kpi-card {p_color}" style="padding:16px">
                <div class="kpi-value" style="font-size:24px">{p_display}</div>
                <div class="kpi-label">P-Value (α={alpha})</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card kpi-orange" style="padding:16px">
                <div class="kpi-value" style="font-size:24px">{dof}</div>
                <div class="kpi-label">Derajat Kebebasan</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

        if p < alpha:
            st.success(f"✅ **Signifikan** — Ada hubungan bermakna antara `{ab_var}` dan stroke (p {'< 0.0001' if p < 0.0001 else f'= {p:.4f}'} < α={alpha})")
            if p < 0.0001:
                st.caption("ℹ️ Bukti statistik sangat kuat — hubungan ini hampir pasti bukan kebetulan.")
        else:
            st.warning(f"⚠️ **Tidak Signifikan** — Tidak cukup bukti hubungan antara `{ab_var}` dan stroke (p = {p:.4f} > α={alpha})")

        col_ct1, col_ct2 = st.columns(2)
        with col_ct1:
            st.markdown("**Tabel Kontingensi (Observed)**")
            st.dataframe(contingency, use_container_width=True)
        with col_ct2:
            st.markdown("**Tabel Expected (jika H₀ benar)**")
            exp_df = pd.DataFrame(expected.round(1), index=contingency.index, columns=contingency.columns)
            st.dataframe(exp_df, use_container_width=True)

        # Visualisasi heatmap contingency
        # Normalized heatmap (persentase per kolom agar perbedaan terlihat jelas)
        cont_norm = contingency.div(contingency.sum(axis=0), axis=1) * 100  # % per kolom
        y_labels  = [str(v) for v in contingency.index.tolist()]
        # Label yang lebih deskriptif
        ab_val_map = {
            "hypertension":    {0:"Tidak Hipertensi", 1:"Hipertensi"},
            "heart_disease":   {0:"Tidak P.Jantung",  1:"P.Jantung"},
            "ever_married":    {"No":"Belum Menikah",  "Yes":"Sudah Menikah"},
            "gender":          {"Male":"Laki-laki",    "Female":"Perempuan", "Other":"Lainnya"},
        }
        y_display = [ab_val_map.get(ab_var, {}).get(v, str(v)) for v in contingency.index.tolist()]

        # Teks annotasi: tampilkan % dan jumlah absolut
        text_annot = [[f"{cont_norm.values[i][j]:.1f}%<br>({contingency.values[i][j]:,})"
                       for j in range(cont_norm.shape[1])]
                      for i in range(cont_norm.shape[0])]

        fig_cont = go.Figure(go.Heatmap(
            z=cont_norm.values,
            x=['Tidak Stroke','Stroke'],
            y=y_display,
            colorscale='RdBu_r',
            zmin=0, zmax=100,
            text=text_annot,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            hovertemplate="%{y} | %{x}: <b>%{z:.1f}%</b><extra></extra>",
            colorbar=dict(title="% dalam<br>kolom", ticksuffix="%")
        ))
        fig_cont.update_layout(
            title="Heatmap Tabel Kontingensi (% per kolom)",
            height=max(280, len(y_labels) * 100),
            **PLOTLY_LAYOUT,
            yaxis=dict(tickmode='array', tickvals=list(range(len(y_display))),
                       ticktext=y_display, color="white", autorange="reversed")
        )
        st.plotly_chart(fig_cont, use_container_width=True)

        # Interpretasi otomatis
        for i, yd in enumerate(y_display):
            pct_nonstroke = cont_norm.values[i][0]
            pct_stroke    = cont_norm.values[i][1]
            n_stroke_val  = contingency.values[i][1]
            n_total_val   = contingency.values[i].sum()
            diff          = pct_stroke - pct_nonstroke
            if abs(diff) > 3:
                arrow = "🔴 lebih tinggi" if diff > 0 else "🟢 lebih rendah"
                st.markdown(f"""<div class="insight-box" style="margin-bottom:8px">
                <p>📌 <b>{yd}</b>: proporsi stroke <b>{arrow}</b> ({pct_stroke:.1f}% vs {pct_nonstroke:.1f}%)
                — dari {n_total_val:,} pasien dengan kondisi ini, <b style="color:#FF4B4B">{n_stroke_val:,} ({pct_stroke:.1f}%) mengalami stroke</b>.</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        **Hipotesis:**
        - **H₀**: Tidak ada hubungan antara variabel tersebut dengan stroke
        - **H₁**: Ada hubungan antara variabel tersebut dengan stroke
        """)

    with tab_cm:
        st.subheader("🔲 Confusion Matrix & Metrik Evaluasi")

        def predict_stroke(row):
            score = row['age']*0.03 + row['avg_glucose_level']*0.01 + row['bmi']*0.01 \
                    + row['hypertension']*2 + row['heart_disease']*2
            return 1 if min(score/10, 1) > 0.5 else 0

        df['y_pred'] = df.apply(predict_stroke, axis=1)
        df['y_pred_label'] = df['y_pred'].map({1:"Stroke", 0:"Tidak Stroke"})

        TP = int(((df['stroke']==1) & (df['y_pred']==1)).sum())
        TN = int(((df['stroke']==0) & (df['y_pred']==0)).sum())
        FP = int(((df['stroke']==0) & (df['y_pred']==1)).sum())
        FN = int(((df['stroke']==1) & (df['y_pred']==0)).sum())
        total = TP+TN+FP+FN
        accuracy  = (TP+TN)/total
        precision = TP/(TP+FP) if (TP+FP)>0 else 0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

        col_cm, col_met = st.columns([3,2])

        with col_cm:
            labels_cm = [
                [f"<b>TRUE POSITIVE</b><br>{TP}<br><i>Stroke → Stroke ✅</i>",
                 f"<b>FALSE NEGATIVE</b><br>{FN}<br><i>Stroke → Tidak ❌</i>"],
                [f"<b>FALSE POSITIVE</b><br>{FP}<br><i>Tidak → Stroke ❌</i>",
                 f"<b>TRUE NEGATIVE</b><br>{TN}<br><i>Tidak → Tidak ✅</i>"]
            ]
            fig_cm = go.Figure()
            cell_colors = [["#2563a8","#a84444"],["#a86c20","#2a7a50"]]
            for i in range(2):
                for j in range(2):
                    fig_cm.add_shape(type="rect",
                        x0=j, x1=j+1, y0=1-i, y1=2-i,
                        fillcolor=cell_colors[i][j], line=dict(color="white", width=2))
                    fig_cm.add_annotation(
                        x=j+0.5, y=1.5-i, text=labels_cm[i][j],
                        showarrow=False, font=dict(color="white", size=13), align="center")
            fig_cm.update_layout(
                title="Confusion Matrix — Prediksi vs Aktual",
                xaxis=dict(tickvals=[0.5,1.5], ticktext=["Prediksi: Stroke","Prediksi: Tidak Stroke"],
                           showgrid=False, zeroline=False, color="white"),
                yaxis=dict(tickvals=[0.5,1.5], ticktext=["Aktual: Tidak Stroke","Aktual: Stroke"],
                           showgrid=False, zeroline=False, color="white"),
                height=340, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_met:
            st.markdown("<br>", unsafe_allow_html=True)
            for lbl, val, clr, hint in [
                ("Akurasi", accuracy, "#00BFFF", "% prediksi yang benar"),
                ("Precision", precision, "#00e676", "Dari yg diprediksi stroke, berapa yang benar"),
                ("Recall", recall, "#FFA500", "Dari kasus stroke nyata, berapa yang terdeteksi"),
                ("F1-Score", f1, "#FF4B4B", "Harmonic mean Precision & Recall")
            ]:
                pct = val*100
                bar_w = int(pct)
                st.markdown(f"""
                <div style="margin-bottom:16px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="color:white;font-weight:600">{lbl}</span>
                    <span style="color:{clr};font-weight:700">{pct:.1f}%</span>
                  </div>
                  <div style="background:#0f1f3d;border-radius:6px;height:10px;overflow:hidden">
                    <div style="background:{clr};width:{bar_w}%;height:100%;border-radius:6px;
                                transition:width 0.5s ease"></div>
                  </div>
                  <div style="color:#90caf9;font-size:11px;margin-top:3px">{hint}</div>
                </div>""", unsafe_allow_html=True)

        st.info("💡 **Recall** lebih penting dari Accuracy untuk deteksi stroke — lebih baik ada *false alarm* daripada melewatkan kasus nyata.")

        st.markdown("---")
        st.subheader("🔍 Detail Prediksi per Pasien")
        filter_pred = st.radio("Tampilkan:", ["Semua","True Positive","True Negative","False Negative (Terlewat!)","False Positive (False Alarm)"], horizontal=True)
        filter_map = {
            "Semua": df,
            "True Positive": df[(df['stroke']==1)&(df['y_pred']==1)],
            "True Negative": df[(df['stroke']==0)&(df['y_pred']==0)],
            "False Negative (Terlewat!)": df[(df['stroke']==1)&(df['y_pred']==0)],
            "False Positive (False Alarm)": df[(df['stroke']==0)&(df['y_pred']==1)],
        }
        show_df = filter_map[filter_pred][['age','hypertension','heart_disease',
                                            'avg_glucose_level','bmi','stroke_label','y_pred_label']].head(20)
        show_df.columns = ['Usia','Hipertensi','P.Jantung','Glukosa','BMI','Aktual','Prediksi']
        st.dataframe(show_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: KESIMPULAN — Enhanced
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📋 Kesimpulan":
    st.markdown("""
    <div class="main-header">
      <h1>📋 Kesimpulan & Rekomendasi</h1>
      <p>Ringkasan temuan utama dari analisis stroke prediction dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card kpi-red"><div class="kpi-value">68</div>
            <div class="kpi-label">Rata-rata Usia Stroke</div><div class="kpi-delta" style="color:#FF4B4B">tahun</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card kpi-orange"><div class="kpi-value">133</div>
            <div class="kpi-label">Rata-rata Glukosa Stroke</div><div class="kpi-delta" style="color:#FFA500">mg/dL</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card" style="background:linear-gradient(135deg,#1a2744,#0f1f3d);border:1px solid #243b6b">
            <div class="kpi-value" style="color:#aa80ff">26.5%</div>
            <div class="kpi-label">Hipertensi pada Stroke</div>
            <div class="kpi-delta" style="color:#aa80ff">vs 9.5% non-stroke</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card kpi-blue"><div class="kpi-value">3</div>
            <div class="kpi-label">Faktor Utama</div><div class="kpi-delta" style="color:#00BFFF">Usia, Glukosa, Hipertensi</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sunburst — faktor risiko
    st.subheader("🌐 Peta Faktor Risiko Stroke")
    risk_data = pd.DataFrame({
        "kategori": ["Faktor Medis"]*5 + ["Gaya Hidup"]*3 + ["Demografis"]*3,
        "subkategori": ["Usia Tua","Hipertensi","Penyakit Jantung","Diabetes","Obesitas",
                        "Merokok","Kurang Gerak","Stres Kronik",
                        "Jenis Kelamin","Status Menikah","Tipe Pekerjaan"],
        "nilai": [40, 30, 25, 35, 15, 20, 10, 10, 8, 5, 7],
        "dampak": ["Sangat Tinggi","Tinggi","Tinggi","Tinggi","Sedang",
                   "Sedang","Sedang","Sedang","Rendah","Rendah","Rendah"]
    })
    fig_sun = px.sunburst(risk_data, path=['kategori','subkategori'], values='nilai',
                           color='nilai', color_continuous_scale='Reds',
                           title="Peta Faktor Risiko Stroke (berdasarkan bobot literatur)")
    fig_sun.update_layout(height=450, **PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("---")

    # Insights
    st.subheader("🔍 Temuan Utama")
    insights = [
        ("🔴", "Usia adalah faktor risiko terkuat",
         "Pasien stroke rata-rata berusia ~68 tahun vs ~41 tahun. Risiko meningkat signifikan pada usia 51+.",
         "danger"),
        ("🟠", "Kadar glukosa tinggi meningkatkan risiko",
         "Pasien stroke rata-rata glukosa ~133 mg/dL vs ~103 mg/dL. Waspadai jika ≥ 126 mg/dL (diabetes).",
         "warn"),
        ("🟡", "Hipertensi berkontribusi signifikan (Chi-Square terbukti)",
         "Proporsi hipertensi pada pasien stroke 26.5% vs 9.5% — terbukti signifikan lewat uji Chi-Square.",
         "warn"),
        ("🟢", "Penyakit jantung memperparah risiko",
         "Pasien dengan riwayat jantung memiliki kemungkinan stroke lebih tinggi.",
         "normal"),
        ("🔵", "BMI berperan namun tidak dominan",
         "BMI berpengaruh tidak langsung melalui risiko hipertensi dan diabetes.",
         "normal"),
        ("🟣", "Status merokok & pekerjaan berpengaruh",
         "'Formerly smoked' dan 'Private worker' memiliki proporsi stroke lebih tinggi dalam dataset.",
         "normal"),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc, cls) in enumerate(insights):
        col = c1 if i%2==0 else c2
        col.markdown(f"""
        <div class="info-card">
          <h4>{icon} {title}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 Rekomendasi Klinis")
    recs = [
        ("🏥", "Skrining Rutin", "Prioritaskan populasi usia >50 tahun dengan hipertensi atau penyakit jantung."),
        ("🍬", "Kelola Glukosa", "Pengelolaan ketat untuk penderita diabetes (target glukosa < 126 mg/dL)."),
        ("🚭", "Gaya Hidup Sehat", "Berhenti merokok, jaga BMI 18.5–24.9, olahraga rutin 150 menit/minggu."),
        ("🤖", "Model Prediksi", "Prioritaskan fitur: age, avg_glucose_level, bmi, hypertension, heart_disease."),
        ("📊", "Optimasi Metrik", "Utamakan Recall dalam evaluasi model stroke — jangan sampai kasus positif terlewat."),
    ]
    for icon, title, desc in recs:
        st.markdown(f"""
        <div style="background:#0f1f3d;border-radius:10px;padding:14px 18px;margin-bottom:10px;
                    border:1px solid #1e3a5f;display:flex;gap:12px;align-items:flex-start">
          <div style="font-size:24px;line-height:1">{icon}</div>
          <div>
            <div style="color:white;font-weight:700;font-size:14px;margin-bottom:4px">{title}</div>
            <div style="color:#cdd6f4;font-size:13px">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Dashboard dibuat untuk keperluan Data Science | Stroke Prediction Dataset (Kaggle - fedesoriano) | v7 — Enhanced Interactive")
