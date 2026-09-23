import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
from simulation import run_monte_carlo_simulation

# 1. Configurazione Pagina
st.set_page_config(
    page_title="Monte Carlo Euro 2024 - Casino Analytics", 
    page_icon="🎰", 
    layout="wide"
)

# Mappatura dei nomi nazioni StatsBomb verso FlagCDN
FLAG_CODES = {
    "Spain": "es", "England": "gb-eng", "France": "fr", "Portugal": "pt",
    "Germany": "de", "Italy": "it", "Netherlands": "nl", "Switzerland": "ch",
    "Turkey": "tr", "Austria": "at", "Belgium": "be", "Denmark": "dk",
    "Slovakia": "sk", "Romania": "ro", "Georgia": "ge", "Slovenia": "si",
    "Albania": "al", "Croatia": "hr", "Czech Republic": "cz", "Poland": "pl",
    "Serbia": "rs", "Scotland": "gb-sct", "Hungary": "hu", "Ukraine": "ua",
    "Spagna": "es", "Inghilterra": "gb-eng", "Francia": "fr", "Portogallo": "pt",
    "Germania": "de", "Italia": "it", "Olanda": "nl", "Svizzera": "ch",
    "Turchia": "tr", "Austria": "at", "Belgio": "be", "Danimarca": "dk",
    "Slovacchia": "sk", "Romania": "ro", "Georgia": "ge", "Slovenia": "si",
    "Albania": "al", "Croazia": "hr", "Repubblica Ceca": "cz", "Polonia": "pl",
    "Serbia": "rs", "Scozia": "gb-sct", "Ungheria": "hu", "Ucraina": "ua"
}

def get_flag_url(team_name):
    code = FLAG_CODES.get(team_name)
    if not code:
        for name, c in FLAG_CODES.items():
            if name.lower() in team_name.lower():
                code = c
                break
    if not code:
        code = "un"
    return f"https://flagcdn.com/w160/{code}.png"

# 2. CSS Personalizzato Monte Carlo Casino
st.markdown("""
    <style>
    /* Fondo Panno Verde Lusso */
    .stApp {
        background: radial-gradient(circle at center, #0f3d23 0%, #05190e 100%);
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    h1, h2, h3 {
        color: #f1c40f !important;
        font-family: 'Georgia', serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    }
    
    /* Header Titolo Stilizzato */
    .title-box {
        border-bottom: 2px solid #d4af37;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    
    /* Style per Pulsanti Fiche */
    div.stButton > button {
        background: linear-gradient(135deg, #c8102e 0%, #8b0000 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border-radius: 30px !important;
        border: 2px solid #f1c40f !important;
        padding: 12px 30px !important;
        box-shadow: 0px 4px 15px rgba(241, 196, 15, 0.4) !important;
        width: 100%;
        cursor: pointer;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(241, 196, 15, 0.7) !important;
    }
    
    /* Cards Fiche */
    .casino-card {
        background: linear-gradient(145deg, rgba(20,40,30,0.95), rgba(10,20,15,0.95));
        border: 2px solid #d4af37;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
    }
    .fiche-title { font-size: 14px; color: #e0e0e0; font-weight: 600; text-transform: uppercase; }
    .fiche-val { font-size: 28px; font-weight: bold; color: #f1c40f; text-shadow: 1px 1px 3px #000; margin: 5px 0; }
    .fiche-sub { font-size: 13px; color: #a0a0a0; }

    /* Flag Image */
    .team-flag {
        height: 38px;
        width: 55px;
        object-fit: cover;
        border-radius: 4px;
        border: 1px solid #d4af37;
        vertical-align: middle;
        margin: 0 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Titolo Principale
st.markdown("""
    <div class="title-box">
        <h1 style="margin: 0;">🎰 Monte Carlo Euro 2024 Simulator</h1>
        <p style="color: #d4af37; margin: 5px 0 0 0;">🎲 Analisi Stocastica Shot-by-Shot • Football Analytics</p>
    </div>
""", unsafe_allow_html=True)

# 4. Caricamento Dataset
@st.cache_data
def load_data():
    with open("data/euro2024_matches.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    dataset = load_data()
except Exception as e:
    st.error("Errore nel caricamento del file `data/euro2024_matches.json`.")
    st.stop()

# 5. Sidebar (Solo Bandiera Monaco)
st.sidebar.image("https://flagcdn.com/w160/mc.png", width=90)
st.sidebar.header("👑 Tavolo da Gioco")

match_keys = list(dataset.keys())
selected_key = st.sidebar.selectbox("Seleziona il Match:", match_keys)

match_data = dataset[selected_key]
info = match_data["match_info"]
shots_a_orig = match_data["shots"]["team_a"]
shots_b_orig = match_data["shots"]["team_b"]

flag_a = get_flag_url(info['team_a'])
flag_b = get_flag_url(info['team_b'])

n_sims = st.sidebar.slider("Numero Simulazioni (Lanci di Dado):", 10000, 200000, 100000, step=10000)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Fase:** `{info['stage']}`")
st.sidebar.markdown(f"**Risultato Reale:** **{info['team_a']} {info['score_real'][0]} - {info['score_real'][1]} {info['team_b']}**")

# Tabs
tab1, tab2 = st.tabs(["📊 Tavolo da Gioco Ufficiale", "🧪 Sandbox 'What If'"])

# ==========================================
# TAB 1: SIMULAZIONE UFFICIALE
# ==========================================
with tab1:
    st.markdown(f"""
        <h2 style='text-align: center; margin-bottom: 20px;'>
            <img src="{flag_a}" class="team-flag"> {info['team_a']} vs {info['team_b']} <img src="{flag_b}" class="team-flag">
        </h2>
    """, unsafe_allow_html=True)
    
    shots_a_xg = [s["xg"] for s in shots_a_orig]
    shots_b_xg = [s["xg"] for s in shots_b_orig]
    
    run_sim = st.button("🎲 GIRA LA ROULETTE (Simula Match)")
    
    if run_sim:
        st.components.v1.html("""
            <audio autoplay preload="auto">
                <source src="https://assets.mixkit.co/active_storage/sfx/2003/2003-preview.mp3" type="audio/mp3">
            </audio>
        """, height=0)
        
    res = run_monte_carlo_simulation(shots_a_xg, shots_b_xg, n_sims)
    
    # Cards Risultati (Fiches)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    c1.markdown(f"""
        <div class='casino-card'>
            <div class='fiche-title'>Vittoria {info['team_a']}</div>
            <div class='fiche-val'>{res['p_win_a']}%</div>
            <div class='fiche-sub'>xPTS: {res['xpts_a']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
        <div class='casino-card'>
            <div class='fiche-title'>Pareggio</div>
            <div class='fiche-val'>{res['p_draw']}%</div>
            <div class='fiche-sub'>Stocastico</div>
        </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
        <div class='casino-card'>
            <div class='fiche-title'>Vittoria {info['team_b']}</div>
            <div class='fiche-val'>{res['p_win_b']}%</div>
            <div class='fiche-sub'>xPTS: {res['xpts_b']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
        <div class='casino-card'>
            <div class='fiche-title'>xG Totali</div>
            <div class='fiche-val'>{res['xg_tot_a']} - {res['xg_tot_b']}</div>
            <div class='fiche-sub'>StatsBomb Data</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Matrice dei Risultati Esatti (%)")
    
    # Heatmap
    z_matrix = res["matrix"]
    x_labels = ["0 Gol", "1 Gol", "2 Gol", "3 Gol", "4 Gol", "5+ Gol"]
    y_labels = ["0 Gol", "1 Gol", "2 Gol", "3 Gol", "4 Gol", "5+ Gol"]
    
    fig = px.imshow(
        z_matrix,
        labels=dict(x=f"Gol {info['team_b']}", y=f"Gol {info['team_a']}", color="Probabilità %"),
        x=x_labels,
        y=y_labels,
        text_auto=".1f",
        color_continuous_scale=[
            [0.0, "#081c15"],
            [0.2, "#1b4332"],
            [0.5, "#d4af37"],
            [1.0, "#c8102e"]
        ]
    )
    
    fig.update_traces(
        textfont=dict(size=14, color="#ffffff", family="Arial Black")
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f1c40f", size=14),
        xaxis=dict(gridcolor="rgba(212,175,55,0.2)", title_font=dict(size=15, color="#f1c40f")),
        yaxis=dict(gridcolor="rgba(212,175,55,0.2)", title_font=dict(size=15, color="#f1c40f")),
        height=480
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SANDBOX "WHAT IF"
# ==========================================
with tab2:
    st.subheader("🧪 Sandbox 'What If' (Stanza Privata del Giocatore)")
    st.write("Modifica gli xG, aggiungi nuovi tiri o elimina occasioni per ricalcolare le probabilità.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"#### Tiri <img src='{flag_a}' class='team-flag' style='height:25px; width:38px;'> {info['team_a']}", unsafe_allow_html=True)
        df_a = pd.DataFrame(shots_a_orig) if shots_a_orig else pd.DataFrame(columns=["player", "minute", "xg"])
        edited_a = st.data_editor(df_a[["player", "minute", "xg"]], num_rows="dynamic", key="edit_a")
        
    with col_b:
        st.markdown(f"#### Tiri <img src='{flag_b}' class='team-flag' style='height:25px; width:38px;'> {info['team_b']}", unsafe_allow_html=True)
        df_b = pd.DataFrame(shots_b_orig) if shots_b_orig else pd.DataFrame(columns=["player", "minute", "xg"])
        edited_b = st.data_editor(df_b[["player", "minute", "xg"]], num_rows="dynamic", key="edit_b")
        
    if st.button("🔄 RICALCOLA SIMULAZIONE WHAT IF"):
        mod_xg_a = edited_a["xg"].dropna().tolist()
        mod_xg_b = edited_b["xg"].dropna().tolist()
        
        res_whatif = run_monte_carlo_simulation(mod_xg_a, mod_xg_b, n_sims)
        
        st.success("✨ Simulazione What If ricalcolata con successo!")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Vittoria {info['team_a']}", f"{res_whatif['p_win_a']}%")
        c2.metric("Pareggio", f"{res_whatif['p_draw']}%")
        c3.metric(f"Vittoria {info['team_b']}", f"{res_whatif['p_win_b']}%")
