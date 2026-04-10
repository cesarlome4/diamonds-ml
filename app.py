import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ── Config page
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)

# ── CSS
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .price-box {
        background: linear-gradient(135deg, #1E293B, #2563EB);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .price-label {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .price-value {
        color: #FFFFFF;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .price-sub {
        color: #93C5FD;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.8rem;
    }
    .info-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .stSlider > div > div { accent-color: #2563EB; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DONNÉES & MODÈLE (généré à la volée, reproductible)
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Entraînement du modèle…")
def train_model():
    np.random.seed(42)
    N = 8000

    # Dataset synthétique fidèle à Diamonds
    carat     = np.random.lognormal(-0.3, 0.5, N).clip(0.2, 5.0)
    cut_cats  = ['Fair','Good','Very Good','Premium','Ideal']
    color_cats= ['J','I','H','G','F','E','D']
    clarity_cats=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

    cut     = np.random.choice(cut_cats,    N, p=np.array([0.03,0.09,0.22,0.26,0.40]))
    color   = np.random.choice(color_cats,  N, p=np.array([0.085,0.111,0.145,0.190,0.173,0.174,0.122]))
    clarity = np.random.choice(clarity_cats,N, p=np.array([0.030,0.171,0.243,0.189,0.146,0.092,0.069,0.060]))

    cut_m = {'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
    col_m = {'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
    cla_m = {'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}

    x = (carat**0.33)*4.5 + np.random.normal(0,0.1,N)
    y = x + np.random.normal(0,0.05,N)
    z = x*0.618 + np.random.normal(0,0.05,N)
    x = x.clip(0.1,10); y=y.clip(0.1,10); z=z.clip(0.1,10)
    depth = (z/((x+y)/2)*100).clip(43,79)
    table = np.random.normal(57.5,2.2,N).clip(43,95)

    cut_n   = np.array([cut_m[c]   for c in cut])
    color_n = np.array([col_m[c]   for c in color])
    clar_n  = np.array([cla_m[c]   for c in clarity])

    price = np.exp(
        8.5 + 1.9*np.log(carat) + 0.10*cut_n + 0.08*color_n + 0.12*clar_n
        + np.random.normal(0,0.12,N)
    ).clip(300, 20000).astype(int)

    volume      = x * y * z
    ratio_dt    = depth / table

    df = pd.DataFrame({
        'carat':carat,'cut':cut,'color':color,'clarity':clarity,
        'depth':depth,'table':table,'x':x,'y':y,'z':z,
        'volume':volume,'ratio_depth_table':ratio_dt,'price':price
    })

    NUM_COLS = ['carat','depth','table','x','y','z','volume','ratio_depth_table']
    CAT_COLS = ['cut','color','clarity']

    cut_ord     = ['Fair','Good','Very Good','Premium','Ideal']
    color_ord   = ['J','I','H','G','F','E','D']
    clarity_ord = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), NUM_COLS),
        ('cat', OrdinalEncoder(categories=[cut_ord,color_ord,clarity_ord]), CAT_COLS),
    ])

    X = df[NUM_COLS + CAT_COLS]
    y_reg = df['price']

    X_proc = preprocessor.fit_transform(X)
    model  = RandomForestRegressor(n_estimators=200, max_depth=15,
                                    random_state=42, n_jobs=-1)
    model.fit(X_proc, y_reg)

    q3 = float(np.percentile(y_reg, 75))
    return model, preprocessor, q3

model, preprocessor, Q3_SEUIL = train_model()


# ══════════════════════════════════════════════════════
# FONCTION DE PRÉDICTION
# ══════════════════════════════════════════════════════
def predict_price(carat, cut, color, clarity, depth, table):
    x = (carat**0.33) * 4.5
    y = x
    z = x * 0.618
    volume   = x * y * z
    ratio_dt = depth / table

    row = pd.DataFrame([{
        'carat': carat, 'depth': depth, 'table': table,
        'x': x, 'y': y, 'z': z,
        'volume': volume, 'ratio_depth_table': ratio_dt,
        'cut': cut, 'color': color, 'clarity': clarity
    }])
    X_proc = preprocessor.transform(row)
    price  = model.predict(X_proc)[0]
    return round(price)


# ══════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════
st.title("💎 Diamond Price Predictor")
st.markdown("Entrez les caractéristiques d'un diamant pour estimer son prix.")
st.divider()

# ── Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⚖️ Caractéristiques physiques")
    carat = st.slider("Carat (poids)", 0.2, 5.0, 1.0, 0.01,
                      help="Plus le carat est élevé, plus le diamant est lourd et cher")
    depth = st.slider("Depth % (profondeur)", 43.0, 79.0, 61.7, 0.1,
                      help="Profondeur totale en % par rapport au diamètre moyen")
    table = st.slider("Table % (largeur table)", 43.0, 95.0, 57.5, 0.5,
                      help="Largeur de la facette supérieure en % du diamètre")

with col2:
    st.markdown("#### 🏆 Qualité (les 3C)")
    cut = st.selectbox("Cut (taille)",
                       ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                       index=4,
                       help="Ideal = meilleure qualité de taille")
    color = st.selectbox("Color (couleur)",
                         ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                         index=6,
                         help="D = incolore (meilleur), J = légèrement coloré")
    clarity = st.selectbox("Clarity (clarté)",
                           ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                           index=7,
                           help="IF = Internally Flawless (meilleur), I1 = inclusions visibles")

st.divider()

# ── Bouton
if st.button("✨ Estimer le prix", use_container_width=True, type="primary"):

    price = predict_price(carat, cut, color, clarity, depth, table)
    is_premium = price >= Q3_SEUIL

    # ── Affichage prix
    grade_html = (
        '<span class="grade-badge" style="background:#D1FAE5;color:#065F46;">⭐ Haut de gamme</span>'
        if is_premium else
        '<span class="grade-badge" style="background:#E0F2FE;color:#0369A1;">✓ Standard</span>'
    )

    st.markdown(f"""
    <div class="price-box">
        <div class="price-label">Prix estimé</div>
        <div class="price-value">${price:,}</div>
        <div class="price-sub">Modèle : Random Forest · Entraîné sur 8 000 diamants</div>
        {grade_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Détail des facteurs
    st.markdown("#### 📊 Détail des facteurs")

    cut_scores    = {'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
    color_scores  = {'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
    clarity_scores= {'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}

    factors = {
        "⚖️ Carat":   (carat / 5.0,           f"{carat} ct"),
        "✂️ Cut":     (cut_scores[cut] / 5,    cut),
        "🎨 Color":   (color_scores[color] / 7, color),
        "🔍 Clarity": (clarity_scores[clarity]/8, clarity),
    }

    for label, (score, val) in factors.items():
        pct = int(score * 100)
        color_bar = "#2563EB" if score > 0.6 else "#D97706" if score > 0.3 else "#DC2626"
        st.markdown(f"""
        <div class="info-card">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-weight:600;color:#1E293B">{label}</span>
                <span style="color:#64748B;font-size:0.9rem">{val} — {pct}%</span>
            </div>
            <div style="background:#E2E8F0;border-radius:999px;height:8px">
                <div style="background:{color_bar};width:{pct}%;height:8px;border-radius:999px;transition:width 0.3s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Conseil
    st.divider()
    if not is_premium and carat < 2.0:
        tip_carat = round(carat + 0.5, 1)
        st.info(f"💡 En augmentant le carat à **{tip_carat}**, le prix estimé serait d'environ "
                f"**${predict_price(tip_carat, cut, color, clarity, depth, table):,}**")
    elif cut != 'Ideal':
        st.info(f"💡 En passant à la taille **Ideal**, le prix estimé serait d'environ "
                f"**${predict_price(carat, 'Ideal', color, clarity, depth, table):,}**")
    else:
        st.success("✅ Ce diamant est déjà dans la configuration optimale pour son carat.")

else:
    st.markdown("""
    <div style="text-align:center;padding:2rem;color:#94A3B8">
        <div style="font-size:3rem">💎</div>
        <div style="margin-top:0.5rem">Configurez les caractéristiques ci-dessus<br>puis cliquez sur <b>Estimer le prix</b></div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer
st.divider()
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:0.8rem'>"
    "Random Forest · MinMaxScaler · OrdinalEncoder · Diamonds Dataset"
    "</p>",
    unsafe_allow_html=True
)
