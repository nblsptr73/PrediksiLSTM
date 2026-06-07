import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard Kunjungan Mixue",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(BASE_DIR, "DATA.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model_lstm_kunjungan.h5")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler_kunjungan.pkl")
FSCALER_PATH = os.path.join(BASE_DIR, "scaler_multi_kunjungan.pkl")

ADMIN_USERS     = {"admin1": "admin123"}


# ═══════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0d1117 0%,#161b27 60%,#0d1117 100%);color:#e6edf3;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent !important;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117 0%,#161b27 100%);border-right:1px solid #21262d;}
[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
[data-testid="stSidebar"] h3{color:#58a6ff !important;}

.hero{background:linear-gradient(135deg,#0e2040 0%,#1a3a6b 50%,#0e2040 100%);border:1px solid #2d4a6b;border-radius:18px;padding:36px 44px;margin-bottom:24px;position:relative;overflow:hidden;}
.hero-tag{display:inline-block;background:rgba(88,166,255,.15);border:1px solid rgba(88,166,255,.3);color:#58a6ff;font-size:.72rem;font-weight:700;padding:4px 12px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;margin-bottom:14px;}
.hero-title{font-size:2.4rem;font-weight:900;line-height:1.15;margin-bottom:10px;background:linear-gradient(90deg,#79c0ff,#a5d6ff,#cae8ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:.92rem;color:#8b949e;margin:0;line-height:1.6;}

.kpi-wrap{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
.kpi{background:linear-gradient(145deg,#161b27,#1c2333);border:1px solid #21262d;border-radius:16px;padding:20px 22px;transition:all .2s;}
.kpi:hover{border-color:#388bfd;transform:translateY(-2px);box-shadow:0 8px 24px rgba(56,139,253,.12);}
.kpi-icon{font-size:1.4rem;margin-bottom:8px;}
.kpi-lbl{font-size:.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.9px;margin-bottom:6px;}
.kpi-val{font-size:2rem;font-weight:800;color:#e6edf3;line-height:1;margin-bottom:5px;}
.kpi-sub{font-size:.78rem;font-weight:500;}
.cg{color:#3fb950;}.cr{color:#f85149;}.cb{color:#58a6ff;}.cm{color:#8b949e;}

.chart-box{background:linear-gradient(145deg,#161b27,#1c2333);border:1px solid #21262d;border-radius:16px;padding:22px 26px;margin-bottom:16px;}
.chart-hdr{font-size:1rem;font-weight:700;color:#e6edf3;margin-bottom:3px;}
.chart-sub{font-size:.78rem;color:#8b949e;}

.adm-card{background:linear-gradient(145deg,#161b27,#1c2333);border:1px solid #21262d;border-radius:14px;padding:18px 22px;margin-bottom:14px;}
.adm-title{font-size:.8rem;font-weight:700;color:#58a6ff;text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;}

.al-i{background:rgba(56,139,253,.1);border:1px solid rgba(56,139,253,.25);border-left:3px solid #388bfd;border-radius:8px;padding:10px 14px;color:#a8c8ff;font-size:.83rem;margin-bottom:12px;}
.al-s{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.25);border-left:3px solid #3fb950;border-radius:8px;padding:10px 14px;color:#a8e6af;font-size:.83rem;margin-bottom:12px;}
.al-w{background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.25);border-left:3px solid #d2993b;border-radius:8px;padding:10px 14px;color:#e3c47a;font-size:.83rem;margin-bottom:12px;}

.stTabs [data-baseweb="tab-list"]{background:#161b27;border-radius:10px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{color:#8b949e;border-radius:8px;font-weight:500;}
.stTabs [aria-selected="true"]{background:rgba(56,139,253,.15)!important;color:#58a6ff!important;}

.stButton>button{background:linear-gradient(135deg,#1f6feb,#388bfd)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(56,139,253,.3)!important;}

.stTextInput input,.stNumberInput input,.stDateInput input{background:#1c2333!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important;}
.stSelectbox>div>div{background:#1c2333!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# MODEL  (cached once, shared)
# ═══════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_and_scalers():
    import tensorflow as tf
    model  = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    fs     = joblib.load(FSCALER_PATH) if os.path.exists(FSCALER_PATH) else None
    return model, scaler, fs


# ═══════════════════════════════════════════════════
# DATA  (cached by version, invalidated on admin save)
# ═══════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(_ver: int):
    df = pd.read_csv(DATA_FILE, sep=";", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns and "tanggal" not in df.columns:
        df = df.rename(columns={"date": "tanggal"})
    df["tanggal"]   = pd.to_datetime(df["tanggal"], dayfirst=True)
    df["pelanggan"] = pd.to_numeric(df["pelanggan"], errors="coerce")
    df["suhu"]      = pd.to_numeric(df["suhu"],      errors="coerce")
    df = df.dropna(subset=["tanggal","pelanggan","suhu"])
    return df.sort_values("tanggal").reset_index(drop=True)


def save_data(df: pd.DataFrame):
    out = df.copy()
    out["tanggal"] = out["tanggal"].dt.strftime("%d/%m/%Y")
    cols = [c for c in ["tanggal","pelanggan","suhu"] if c in out.columns]
    out[cols].to_csv(DATA_FILE, sep=";", index=False, encoding="utf-8")
    st.session_state["_dv"] = st.session_state.get("_dv", 0) + 1
    load_data.clear()


# ═══════════════════════════════════════════════════
# FEATURE HELPERS
# ═══════════════════════════════════════════════════
def make_feat_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build (N, 4) feature matrix: [pelanggan, is_weekend, suhu, tgl_angka]."""
    dates      = df["tanggal"]
    is_weekend = dates.dt.weekday.ge(5).astype(float).values
    tgl_angka  = dates.dt.day.astype(float).values
    pelanggan  = df["pelanggan"].astype(float).values
    suhu       = df["suhu"].astype(float).values
    return np.stack([pelanggan, is_weekend, suhu, tgl_angka], axis=1)  # (N,4)


def scale_features(X: np.ndarray, fscaler) -> np.ndarray:
    if fscaler is not None:
        return fscaler.transform(X)
    Xs = (X - np.array([0, 0, -10, 1])) / np.array([1000, 1, 50, 31])
    return np.clip(Xs, 0, 1)


# ═══════════════════════════════════════════════════
# BATCH PREDICTIONS  (one model.predict call!)
# ═══════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_predictions(_ver: int, future_days: int):
    df             = load_data(_ver)
    model, sc, fs  = load_model_and_scalers()
    n              = len(df)
    feat_raw       = make_feat_matrix(df)           # (N, 4)
    feat_scaled    = scale_features(feat_raw, fs)   # (N, 4)

    # ── Historical: build all sliding windows at once ──
    win = 7
    num_wins = n - win + 1  # number of windows
    # Shape: (num_wins, 7, 4)
    windows_hist = np.stack([feat_scaled[i:i+win] for i in range(num_wins)], axis=0)
    # Single batch predict
    pred_scaled_hist = model.predict(windows_hist, verbose=0, batch_size=64)  # (num_wins, 1)
    preds_hist = sc.inverse_transform(pred_scaled_hist.reshape(-1, 1)).flatten()
    preds_hist = np.round(preds_hist).astype(int)

    dates_hist   = df["tanggal"].iloc[win-1:].values
    actuals_hist = df["pelanggan"].iloc[win-1:].astype(int).values

    df_pred = pd.DataFrame({
        "tanggal":  pd.to_datetime(dates_hist),
        "aktual":   actuals_hist,
        "prediksi": preds_hist,
    })
    df_pred["error"]     = df_pred["prediksi"] - df_pred["aktual"]
    df_pred["abs_error"] = df_pred["error"].abs()

    # ── Future forecast ──
    last_suhu  = float(df["suhu"].iloc[-1])
    last_date  = pd.Timestamp(df["tanggal"].iloc[-1])
    fut_win    = feat_scaled[-win:].copy()   # (7,4)
    fut_raw_win= feat_raw[-win:].copy()

    future_dates, future_preds = [], []
    for d in range(1, future_days + 1):
        inp  = fut_win.reshape(1, win, 4)
        ps   = model.predict(inp, verbose=0)
        pval = int(round(float(sc.inverse_transform(ps.reshape(-1,1))[0,0])))
        nd   = last_date + timedelta(days=d)
        future_dates.append(nd)
        future_preds.append(pval)
        # Build next raw row for this future date
        ndt   = nd
        is_we = 1.0 if ndt.weekday() >= 5 else 0.0
        nraw  = np.array([[pval, is_we, last_suhu, float(ndt.day)]])
        nsc   = scale_features(nraw, fs)
        fut_win = np.vstack([fut_win[1:], nsc])

    df_future = pd.DataFrame({"tanggal": future_dates, "forecast": future_preds})
    return df_pred, df_future


# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════
for k, v in [("_dv",0),("admin_in",False),("admin_user",""),
             ("show_login",False),("adm_msg",None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 20px 0;'>
        <div style='font-size:2.6rem;'>🧊</div>
        <div style='font-size:1.1rem;font-weight:800;color:#58a6ff;margin-top:6px;'>Mixue Predictor</div>
        <div style='font-size:.73rem;color:#484f58;margin-top:4px;'>Sistem Prediksi Kunjungan</div>
    </div>
    <hr style='border-color:#21262d;margin:0 0 18px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### 📅 Rentang Grafik")
    day_range = st.slider("Tampilkan data (hari terakhir)", 7, 365, 30, 7)

    st.markdown("#### 🔮 Prediksi ke Depan")
    future_days = st.slider("Berapa hari ke depan?", 1, 14, 7, 1)

    st.markdown("<hr style='border-color:#21262d;margin:14px 0;'>", unsafe_allow_html=True)

    # ── Admin section ──
    if st.session_state["admin_in"]:
        st.markdown(f"""
        <div class='al-s' style='text-align:center;'>
            🛡️ Admin: <strong>{st.session_state['admin_user']}</strong>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", key="btn_lo", use_container_width=True):
            st.session_state.update({"admin_in":False,"admin_user":"",
                                     "show_login":False,"adm_msg":None})
            st.rerun()
    else:
        if not st.session_state["show_login"]:
            if st.button("🔐 Login Admin", key="btn_sl", use_container_width=True):
                st.session_state["show_login"] = True
                st.rerun()
        else:
            st.markdown("**🔐 Login Admin**")
            u = st.text_input("Username", key="lu", placeholder="Username")
            p = st.text_input("Password", type="password", key="lp", placeholder="Password")
            ca, cb = st.columns(2)
            with ca:
                if st.button("Masuk", key="btn_in", use_container_width=True):
                    if ADMIN_USERS.get(u) == p and p != "":
                        st.session_state.update({"admin_in":True,"admin_user":u,
                                                 "show_login":False,"adm_msg":None})
                        st.rerun()
                    else:
                        st.session_state["adm_msg"] = ("err","❌ Username / password salah.")
                        st.rerun()
            with cb:
                if st.button("Batal", key="btn_cl", use_container_width=True):
                    st.session_state.update({"show_login":False,"adm_msg":None})
                    st.rerun()
            if st.session_state["adm_msg"]:
                t, m = st.session_state["adm_msg"]
                cls = "al-s" if t == "ok" else "al-w"
                st.markdown(f"<div class='{cls}'>{m}</div>", unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#21262d;margin:14px 0;'>
    <div style='text-align:center;font-size:.7rem;color:#30363d;'>
        LSTM · TensorFlow · Streamlit<br>© 2025 KP Semester 6
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# LOAD MODEL & DATA  (with status)
# ═══════════════════════════════════════════════════
ver = st.session_state["_dv"]

with st.spinner("⏳ Sedang memuat sistem prediksi…"):
    try:
        _m, _s, _fs = load_model_and_scalers()
        model_ok = True
    except Exception as e:
        model_ok = False; model_err = str(e)

try:
    df_data = load_data(ver)
    data_ok = True
except Exception as e:
    data_ok = False; data_err = str(e)


# ═══════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════
n_data   = len(df_data) if data_ok else 0
d_min    = df_data["tanggal"].min().strftime("%d %b %Y") if data_ok and n_data > 0 else "-"
d_max    = df_data["tanggal"].max().strftime("%d %b %Y") if data_ok and n_data > 0 else "-"

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">🧊 Live Dashboard</div>
    <div class="hero-title">Dashboard Kunjungan Pelanggan</div>
    <p class="hero-sub">
        Prediksi jumlah pelanggan harian secara otomatis menggunakan kecerdasan buatan<br>
        <span style='color:#58a6ff;font-weight:600;'>{n_data} hari data</span>
        &nbsp;({d_min} – {d_max})
    </p>
</div>
""", unsafe_allow_html=True)


# ── Error guards ──
if not data_ok:
    st.error(f"❌ Data kunjungan tidak dapat dibaca. Hubungi admin untuk bantuan.")
    st.stop()
if not model_ok:
    st.warning(f"⚠️ Sistem prediksi belum siap. Pastikan file model tersedia.")
    st.stop()
if n_data < 7:
    st.warning(f"⚠️ Data terlalu sedikit. Dibutuhkan minimal 7 hari data, saat ini hanya {n_data} hari.")
    st.stop()


# ═══════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════
if st.session_state["admin_in"]:
    st.markdown("---")
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>
        <div style='font-size:1.4rem;'>🛡️</div>
        <div>
            <div style='font-size:1.1rem;font-weight:800;color:#e6edf3;'>Panel Admin</div>
            <div style='font-size:.78rem;color:#8b949e;'>Kelola data kunjungan pelanggan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show admin notification
    if st.session_state["adm_msg"] and st.session_state["adm_msg"][0] in ("ok","err"):
        t, m = st.session_state["adm_msg"]
        st.markdown(f"<div class='{'al-s' if t=='ok' else 'al-w'}'>{m}</div>", unsafe_allow_html=True)
        st.session_state["adm_msg"] = None

    t1, t2, t3 = st.tabs(["➕ Tambah Data", "✏️ Edit / Hapus", "📋 Semua Data"])

    # ── Tab 1: Tambah ────────────────────────────────
    with t1:
        st.markdown("<div class='adm-card'><div class='adm-title'>📝 Tambah Entri Baru</div>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1: nd  = st.date_input("📅 Tanggal", value=datetime.today(), key="nd")
        with a2: np_ = st.number_input("👥 Pelanggan", 0, 9999, 150, key="np")
        with a3: ns  = st.number_input("🌡️ Suhu (°C)", 20.0, 45.0, 31.0, 0.5, key="ns")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("💾 Simpan Data Baru", key="btn_add"):
            df_cur = load_data(ver)
            nts    = pd.Timestamp(nd)
            if nts in df_cur["tanggal"].values:
                st.markdown("<div class='al-w'>⚠️ Tanggal sudah ada. Gunakan tab Edit / Hapus.</div>", unsafe_allow_html=True)
            else:
                new_row = pd.DataFrame([{"tanggal":nts,"pelanggan":np_,"suhu":ns}])
                df_upd  = pd.concat([df_cur, new_row]).sort_values("tanggal").reset_index(drop=True)
                save_data(df_upd)
                compute_predictions.clear()
                st.session_state["adm_msg"] = ("ok", f"✅ Data {nd.strftime('%d/%m/%Y')} — {np_} pelanggan berhasil ditambahkan!")
                st.rerun()

    # ── Tab 2: Edit / Hapus ──────────────────────────
    with t2:
        df_cur = load_data(ver)
        df_cur["tgl_str"] = df_cur["tanggal"].dt.strftime("%d/%m/%Y")
        sel = st.selectbox("Pilih tanggal:", df_cur["tgl_str"].tolist()[::-1], key="sel_e")

        if sel:
            row = df_cur[df_cur["tgl_str"] == sel].iloc[0]
            st.markdown("<div class='adm-card'>", unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1: ep = st.number_input("👥 Pelanggan", value=int(row["pelanggan"]), key="ep")
            with e2: es = st.number_input("🌡️ Suhu", value=float(row["suhu"]), step=0.5, key="es")
            st.markdown("</div>", unsafe_allow_html=True)

            b1,b2,_ = st.columns([1,1,3])
            with b1:
                if st.button("💾 Update", key="btn_upd", use_container_width=True):
                    idx = df_cur[df_cur["tgl_str"]==sel].index[0]
                    df_cur.at[idx,"pelanggan"] = ep
                    df_cur.at[idx,"suhu"]      = es
                    save_data(df_cur.drop(columns=["tgl_str"]))
                    compute_predictions.clear()
                    st.session_state["adm_msg"] = ("ok", f"✅ Data {sel} berhasil diperbarui!")
                    st.rerun()
            with b2:
                if st.button("🗑️ Hapus", key="btn_del", use_container_width=True):
                    df_upd = df_cur[df_cur["tgl_str"]!=sel].drop(columns=["tgl_str"]).reset_index(drop=True)
                    save_data(df_upd)
                    compute_predictions.clear()
                    st.session_state["adm_msg"] = ("ok", f"✅ Data {sel} berhasil dihapus!")
                    st.rerun()

    # ── Tab 3: Lihat Semua ───────────────────────────
    with t3:
        df_all = load_data(ver).copy()
        # Sort values while 'tanggal' is still a datetime object
        df_all = df_all.sort_values("tanggal", ascending=False).reset_index(drop=True)
        # Then convert to string for display
        df_all["tanggal"] = df_all["tanggal"].dt.strftime("%d/%m/%Y")
        st.markdown(f"<div class='al-i'>📊 Total: <strong>{len(df_all)}</strong> entri</div>", unsafe_allow_html=True)
        st.dataframe(df_all, use_container_width=True, height=360, hide_index=True)

    st.markdown("---")


# ═══════════════════════════════════════════════════
# COMPUTE PREDICTIONS  (batched, fast!)
# ═══════════════════════════════════════════════════
with st.spinner("🔄 Sedang menghitung prediksi, harap tunggu…"):
    try:
        df_pred, df_future = compute_predictions(ver, future_days)
        pred_ok = True
    except Exception as ex:
        pred_ok = False; pred_err = str(ex)

if not pred_ok:
    st.error(f"❌ Gagal menghitung prediksi. Coba muat ulang halaman atau hubungi admin.")
    st.stop()


# ═══════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════
mae         = df_pred["abs_error"].mean()
mape_val    = (df_pred["abs_error"] / df_pred["aktual"].replace(0, np.nan)).mean() * 100
last_actual = int(df_pred["aktual"].iloc[-1])
next_pred   = int(df_future["forecast"].iloc[0]) if len(df_future) else int(df_pred["prediksi"].iloc[-1])
delta_pct   = ((next_pred - last_actual) / max(last_actual, 1)) * 100
dclass      = "cg" if delta_pct >= 0 else "cr"
darr        = "▲" if delta_pct >= 0 else "▼"

st.markdown(f"""
<div class="kpi-wrap">
  <div class="kpi">
    <div class="kpi-icon">🔮</div>
    <div class="kpi-lbl">Prediksi Besok</div>
    <div class="kpi-val">{next_pred:,}</div>
    <div class="kpi-sub {dclass}">{darr} {abs(delta_pct):.1f}% vs hari ini ({last_actual:,})</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">📈</div>
    <div class="kpi-lbl">Rata-rata Harian</div>
    <div class="kpi-val">{df_pred['aktual'].mean():.0f}</div>
    <div class="kpi-sub cm">dari {len(df_pred):,} hari</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-lbl">Rata-rata Kesalahan</div>
    <div class="kpi-val">{mae:.1f}</div>
    <div class="kpi-sub {'cg' if mae<20 else 'cb' if mae<40 else 'cr'}">{'Sangat akurat' if mae<20 else 'Cukup akurat' if mae<40 else 'Perlu peningkatan'}</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">📊</div>
    <div class="kpi-lbl">Tingkat Kebenaran</div>
    <div class="kpi-val">{"N/A" if np.isnan(mape_val) else f"{100-mape_val:.1f}%"}</div>
    <div class="kpi-sub cm">{"" if np.isnan(mape_val) else f"rata-rata meleset {mape_val:.1f}%"}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# LINE CHART
# ═══════════════════════════════════════════════════
import plotly.graph_objects as go

cutoff   = df_pred["tanggal"].max() - pd.Timedelta(days=day_range)
df_chart = df_pred[df_pred["tanggal"] >= cutoff].copy()

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=df_chart["tanggal"], y=df_chart["aktual"],
    mode="lines+markers", name="Aktual",
    line=dict(color="#3fb950", width=2.5),
    marker=dict(size=4, color="#3fb950"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Aktual: <b>%{y:,}</b><extra></extra>",
))
fig.add_trace(go.Scatter(
    x=df_chart["tanggal"], y=df_chart["prediksi"],
    mode="lines+markers", name="Prediksi LSTM",
    line=dict(color="#58a6ff", width=2.5, dash="dot"),
    marker=dict(size=4, color="#58a6ff", symbol="diamond"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Prediksi: <b>%{y:,}</b><extra></extra>",
))

fx = [df_pred["tanggal"].max()] + list(df_future["tanggal"])
fy = [int(df_pred["prediksi"].iloc[-1])] + list(df_future["forecast"])
fig.add_trace(go.Scatter(
    x=fx, y=fy, mode="lines+markers",
    name=f"Prediksi {future_days} Hari ke Depan",
    line=dict(color="#f78166", width=2.5, dash="dash"),
    marker=dict(size=7, color="#f78166", symbol="star"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Prediksi ke Depan: <b>%{y:,}</b><extra></extra>",
))
fig.add_vline(
    x=df_pred["tanggal"].max().timestamp()*1000,
    line_dash="dash", line_color="rgba(255,255,255,0.12)", line_width=1.5,
    annotation_text="Sekarang ▶", annotation_font_color="#8b949e", annotation_font_size=10,
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c9d1d9"),
    legend=dict(bgcolor="rgba(22,27,39,.85)", bordercolor="#21262d", borderwidth=1,
                font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False, title="Jumlah Pelanggan", tickfont=dict(size=10)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#161b27", bordercolor="#21262d", font=dict(color="#e6edf3", size=12)),
    margin=dict(l=0,r=0,t=24,b=0), height=420,
)

st.markdown(f"""
<div class="chart-box">
    <div class="chart-hdr">📈 Grafik Kunjungan — {day_range} Hari Terakhir + Prediksi {future_days} Hari ke Depan</div>
    <div class="chart-sub">Kunjungan nyata (hijau) · Hasil prediksi model (biru) · Prediksi ke depan (oranye)</div>
</div>
""", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════
# DETAIL TABS
# ═══════════════════════════════════════════════════
ta, tb, tc = st.tabs(["📋 Tabel Prediksi", "🔮 Prediksi ke Depan", "📊 Analisis Kesalahan"])

with ta:
    df_tbl = df_chart[["tanggal","aktual","prediksi","error","abs_error"]].copy()
    df_tbl["tanggal"] = df_tbl["tanggal"].dt.strftime("%d %b %Y")
    df_tbl.columns    = ["Tanggal","Kunjungan Nyata","Hasil Prediksi","Selisih","Selisih Mutlak"]
    df_tbl_sorted = df_tbl.sort_values("Tanggal", ascending=False).reset_index(drop=True)

    def warnai_selisih(val):
        maks = df_tbl_sorted["Selisih Mutlak"].max() or 1
        intens = min(val / maks, 1.0)
        g = int(255 * (1 - intens * 0.7))
        return f"background-color: rgba(255,{g},{g},0.35)"

    st.dataframe(
        df_tbl_sorted
              .style.format({"Kunjungan Nyata":"{:,}","Hasil Prediksi":"{:,}","Selisih":"{:+,}","Selisih Mutlak":"{:,}"})
              .map(warnai_selisih, subset=["Selisih Mutlak"]),
        use_container_width=True, height=340, hide_index=True,
    )


with tb:
    dfs = df_future.copy()
    dfs["Hari"] = dfs["tanggal"].dt.day_name().map(
        {"Monday":"Senin","Tuesday":"Selasa","Wednesday":"Rabu","Thursday":"Kamis",
         "Friday":"Jumat","Saturday":"Sabtu","Sunday":"Minggu"})
    dfs["Status"] = dfs["Hari"].apply(lambda x: "🏖️ Akhir Pekan" if x in ["Sabtu","Minggu"] else "💼 Hari Kerja")
    dfs["Tanggal"]  = dfs["tanggal"].dt.strftime("%d %b %Y")
    dfs = dfs.rename(columns={"forecast":"Prediksi"})[["Tanggal","Hari","Status","Prediksi"]]

    fb = go.Figure(go.Bar(
        x=dfs["Tanggal"], y=dfs["Prediksi"],
        marker=dict(color=dfs["Prediksi"], colorscale=[[0,"#0e2040"],[.5,"#1f6feb"],[1,"#79c0ff"]], showscale=False),
        text=dfs["Prediksi"], textposition="outside", textfont=dict(color="#c9d1d9",size=11),
        hovertemplate="<b>%{x}</b><br>Prediksi Kunjungan: <b>%{y:,}</b><extra></extra>",
    ))
    fb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Inter",color="#c9d1d9"),
                     xaxis=dict(gridcolor="#21262d",tickfont=dict(size=10)),
                     yaxis=dict(gridcolor="#21262d",title="Prediksi"),
                     margin=dict(l=0,r=0,t=20,b=0), height=250)
    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar":False})
    st.dataframe(dfs.style.format({"Prediksi":"{:,}"}), use_container_width=True, hide_index=True)

with tc:
    cl, cr = st.columns(2)
    with cl:
        fh = go.Figure(go.Histogram(
            x=df_pred["error"], nbinsx=20,
            marker=dict(color="rgba(88,166,255,.55)", line=dict(color="#388bfd",width=1)),
            hovertemplate="Kesalahan: <b>%{x}</b><br>Frekuensi: <b>%{y}</b><extra></extra>",
        ))
        fh.add_vline(x=0, line_dash="dash", line_color="#3fb950", line_width=1.5,
                     annotation_text="Sempurna", annotation_font_color="#3fb950", annotation_font_size=10)
        fh.update_layout(title=dict(text="Sebaran Kesalahan Prediksi",font=dict(color="#e6edf3",size=12)),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter",color="#c9d1d9"),
                         xaxis=dict(gridcolor="#21262d",title="Selisih Prediksi vs Nyata"),
                         yaxis=dict(gridcolor="#21262d",title="Berapa Kali Terjadi"),
                         margin=dict(l=0,r=0,t=36,b=0), height=260, bargap=.06)
        st.plotly_chart(fh, use_container_width=True, config={"displayModeBar":False})
    with cr:
        fe = go.Figure(go.Scatter(
            x=df_pred["tanggal"], y=df_pred["error"],
            mode="lines", fill="tozeroy", fillcolor="rgba(88,166,255,.07)",
            line=dict(color="#58a6ff",width=1.5),
            hovertemplate="<b>%{x|%d %b}</b><br>Kesalahan: <b>%{y:+}</b><extra></extra>",
        ))
        fe.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        fe.update_layout(title=dict(text="Kesalahan Prediksi per Hari",font=dict(color="#e6edf3",size=12)),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter",color="#c9d1d9"),
                         xaxis=dict(gridcolor="#21262d"),
                         yaxis=dict(gridcolor="#21262d",title="Selisih Prediksi"),
                         margin=dict(l=0,r=0,t=36,b=0), height=260)
        st.plotly_chart(fe, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='al-i'>💡 <strong>Cara membaca:</strong> Angka di bawah menunjukkan seberapa jauh prediksi meleset dari kunjungan nyata. Semakin kecil, semakin bagus.</div>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Rata-rata Meleset",  f"{df_pred['error'].mean():.2f}")
    s2.metric("Variasi Kesalahan",  f"{df_pred['error'].std():.2f}")
    s3.metric("Paling Kurang Prediksi", f"{df_pred['error'].min():+d}")
    s4.metric("Paling Lebih Prediksi",  f"{df_pred['error'].max():+d}")
