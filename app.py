import streamlit as st
import pandas as pd
import altair as alt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Padel Ranking", layout="wide")

# CONSTANTES
STARTING_ELO = {"Débutant": 1000, "Intermédiaire": 1200, "Expert": 1400}
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- CONNEXION GOOGLE SHEETS ---
def get_db_connection():
    """Connecte au Google Sheet via les secrets Streamlit."""
    try:
        # On récupère les infos secrètes configurées dans Streamlit Cloud
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        # Ouvre le fichier par son nom EXACT
        sheet = client.open("Padel_DB")
        return sheet
    except Exception as e:
        st.error(f"Erreur de connexion Google Sheets : {e}")
        st.stop()

# --- FONCTIONS CRUD (Create, Read, Update, Delete) ---
def load_data():
    sheet = get_db_connection()
    # On lit tout d'un coup pour éviter trop d'appels
    players_data = sheet.worksheet("players").get_all_records()
    matches_data = sheet.worksheet("matches").get_all_records()
    return players_data, matches_data, sheet

def add_match_to_db(sheet, match_dict):
    ws = sheet.worksheet("matches")
    # On ajoute une ligne à la fin
    ws.append_row([
        match_dict['date'], match_dict['p1'], match_dict['p2'], 
        match_dict['p3'], match_dict['p4'], match_dict['score'], match_dict['winner']
    ])

def delete_match_from_db(sheet, row_index):
    ws = sheet.worksheet("matches")
    # +2 car gspread commence à 1 et il y a une ligne d'entête
    ws.delete_rows(row_index + 2)

def add_player_to_db(sheet, name, level):
    ws = sheet.worksheet("players")
    ws.append_row([name, level])

def update_player_level_db(sheet, name, new_level):
    ws = sheet.worksheet("players")
    # On cherche la cellule qui contient le nom
    cell = ws.find(name)
    # On met à jour la colonne d'à côté (colonne 2 = level)
    ws.update_cell(cell.row, 2, new_level)

def delete_player_db(sheet, name):
    ws = sheet.worksheet("players")
    cell = ws.find(name)
    ws.delete_rows(cell.row)

# --- LOGIQUE ELO ---
def calculate_elo(rating_a, rating_b, actual_score, k=32):
    expected_score = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return rating_a + k * (actual_score - expected_score)

def process_data(players_list, matches_list):
    # --- CORRECTION : SI LISTE VIDE, RETOURNER VIDE ---
    if not players_list:
        # On retourne un DataFrame vide mais avec les bonnes colonnes pour éviter le crash
        empty_df = pd.DataFrame(columns=['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D'])
        return empty_df, pd.DataFrame(), []
    # --------------------------------------------------

    # 1. Init Joueurs
    players = {}
    for p in players_list:
        # Sécurité si la colonne level est vide
        lvl = p.get('level', 'Intermédiaire') 
        if not lvl: lvl = 'Intermédiaire'
            
        players[p['name']] = {
            'level': lvl,
            'elo': STARTING_ELO.get(lvl, 1200),
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0
        }

    history_records = []
    # Init Graph
    for p_name, data in players.items():
        history_records.append({'Date': 'Début', 'Joueur': p_name, 'ELO': data['elo']})

    # 2. Rejouer Matchs
    processed_matches = []
    
    for idx, m in enumerate(matches_list):
        t1, t2 = [m['p1'], m['p2']], [m['p3'], m['p4']]
        
        # Ignorer si joueurs inconnus
        if not all(p in players for p in t1 + t2): continue
        
        # Sauvegarde pour affichage
        m_display = m.copy()
        m_display['id'] = idx 
        processed_matches.append(m_display)

        avg_t1 = (players[t1[0]]['elo'] + players[t1[1]]['elo']) / 2
        avg_t2 = (players[t2[0]]['elo'] + players[t2[1]]['elo']) / 2
        
        res_t1 = 1 if m['winner'] == 'Team 1' else (0 if m['winner'] == 'Team 2' else 0.5)
        diff = calculate_elo(avg_t1, avg_t2, res_t1) - avg_t1
        
        active_players = t1 + t2
        for p in t1:
            players[p]['elo'] += diff
            players[p]['matches'] += 1
            if res_t1 == 1: players[p]['wins'] += 1
            elif res_t1 == 0.5: players[p]['draws'] += 1
            else: players[p]['losses'] += 1
        for p in t2:
            players[p]['elo'] -= diff
            players[p]['matches'] += 1
            if res_t1 == 0: players[p]['wins'] += 1
            elif res_t1 == 0.5: players[p]['draws'] += 1
            else: players[p]['losses'] += 1
            
        for p in active_players:
            history_records.append({'Date': m['date'], 'Joueur': p, 'ELO': players[p]['elo']})

    # Création du DF final
    df_rank = pd.DataFrame.from_dict(players, orient='index').reset_index()
    df_rank.columns = ['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D']
    df_rank['ELO'] = df_rank['ELO'].round(0).astype(int)
    
    return df_rank.sort_values(by='ELO', ascending=False), pd.DataFrame(history_records), processed_matches

# --- INTERFACE ---
st.title("🎾 Padel League (Cloud Version)")

# Login
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2643/2643479.png", width=50)
    pwd = st.text_input("Mot de passe Admin", type="password")
    IS_ADMIN = (pwd == st.secrets.get("ADMIN_PASSWORD", "padel2024"))
    if IS_ADMIN: st.success("Mode Admin")

# Chargement (Mise en cache pour éviter la lenteur Google)
# Note: On ne cache pas tout ici pour voir les modifs en direct, 
# mais en prod on pourrait utiliser @st.cache_data
raw_players, raw_matches, sheet_conn = load_data()
df_rank, df_hist, list_matches = process_data(raw_players, raw_matches)

tab_names = ["🏆 Classement", "📈 Progression", "⚖️ Générateur"]
if IS_ADMIN: tab_names += ["➕ Saisir Match", "⚙️ Admin"]
tabs = st.tabs(tab_names)

# 1. CLASSEMENT
with tabs[0]:
    if not df_rank.empty:
        st.dataframe(df_rank[['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D']].style.background_gradient(subset=['ELO'], cmap="Greens"), use_container_width=True, hide_index=True)
    else: st.info("Aucun joueur.")

# 2. PROGRESSION
with tabs[1]:
    if not df_hist.empty:
        sel = st.multiselect("Filtrer", df_rank['Joueur'].tolist(), default=df_rank['Joueur'].tolist()[:5])
        if sel:
            sub = df_hist[df_hist['Joueur'].isin(sel)]
            y_dom = [sub['ELO'].min()-20, sub['ELO'].max()+20]
            c = alt.Chart(sub).mark_line(point=True).encode(x='Date', y=alt.Y('ELO', scale=alt.Scale(domain=y_dom)), color='Joueur', tooltip=['Date','Joueur','ELO']).interactive()
            st.altair_chart(c, use_container_width=True)

# 3. GENERATEUR
with tabs[2]:
    pres = st.multiselect("4 Joueurs", df_rank['Joueur'].tolist())
    if len(pres) == 4:
        elos = {n: df_rank[df_rank['Joueur']==n]['ELO'].values[0] for n in pres}
        p = pres
        scens = [((p[0], p[1]), (p[2], p[3])), ((p[0], p[2]), (p[1], p[3])), ((p[0], p[3]), (p[1], p[2]))]
        res = sorted([{'Txt': f"{ta[0]}/{ta[1]} vs {tb[0]}/{tb[1]}", 'Diff': abs((elos[ta[0]]+elos[ta[1]])/2 - (elos[tb[0]]+elos[tb[1]])/2)} for ta, tb in scens], key=lambda x: x['Diff'])
        st.success(f"Best: {res[0]['Txt']} (Diff: {res[0]['Diff']:.1f})")
        for r in res[1:]: st.write(f"{r['Txt']} (Diff: {r['Diff']:.1f})")

if IS_ADMIN:
    # 4. SAISIE
    with tabs[3]:
        st.header("Saisie Match")
        lst = [p['name'] for p in raw_players]
        c1, _, c3 = st.columns([1,0.2,1])
        with c1: 
            p1 = st.selectbox("J1", lst); p2 = st.selectbox("J2", [x for x in lst if x!=p1])
        with c3:
            rem = [x for x in lst if x not in [p1,p2]]
            p3 = st.selectbox("J3", rem); p4 = st.selectbox("J4", [x for x in rem if x!=p3])
        
        sc = st.text_input("Score")
        da = st.date_input("Date", datetime.now())
        res = st.radio("Résultat", ["Team 1", "Team 2", "Egalité"], horizontal=True)
        winner = "Team 1" if res == "Team 1" else ("Team 2" if res == "Team 2" else "Draw")
        
        if st.button("Valider"):
            add_match_to_db(sheet_conn, {'date': str(da), 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'score': sc, 'winner': winner})
            st.success("Enregistré sur Google Sheets !")
            st.rerun()

    # 5. ADMIN
    with tabs[4]:
        st.header("Admin")
        with st.expander("Ajout Joueur"):
            n = st.text_input("Nom"); l = st.selectbox("Niveau", ["Débutant", "Intermédiaire", "Expert"], index=1)
            if st.button("Créer"):
                add_player_to_db(sheet_conn, n, l)
                st.rerun()
        
        with st.expander("Modif Niveau"):
            who = st.selectbox("Joueur", [p['name'] for p in raw_players])
            new_l = st.selectbox("Nouveau Niveau", ["Débutant", "Intermédiaire", "Expert"], key="nl")
            if st.button("Mettre à jour"):
                update_player_level_db(sheet_conn, who, new_l)
                st.rerun()
                
        st.divider()
        st.subheader("Suppression Match")
        # On affiche la liste inversée
        if list_matches:
            to_del = st.selectbox("Match à supprimer", [m['id'] for m in list_matches][::-1], format_func=lambda x: f"{list_matches[x]['date']} - {list_matches[x]['p1']}...")
            if st.button("Supprimer ce match"):
                delete_match_from_db(sheet_conn, to_del)
                st.success("Supprimé !")
                st.rerun()
