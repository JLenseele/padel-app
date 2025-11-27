import streamlit as st
import pandas as pd
import altair as alt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURATION & CONSTANTES ---
st.set_page_config(page_title="Sport ELO Manager", layout="wide")

# Paramètres du sport
SPORTS_CONFIG = {
    "padel": {
        "name": "🎾 Padel",
        "team_size": 2, # 2 joueurs par équipe (4 au total)
        "players_sheet": "padel_players",
        "matches_sheet": "padel_matches"
    },
    "futsal": {
        "name": "⚽ Futsal",
        # Note: Nous gérons la taille de l'équipe dynamiquement au Futsal (4, 5 ou 6)
        "team_size": 4, 
        "players_sheet": "futsal_players",
        "matches_sheet": "futsal_matches"
    }
}

STARTING_ELO = {"Débutant": 1000, "Intermédiaire": 1200, "Expert": 1400}
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- FONCTIONS BASE DE DONNÉES (MODULARISÉES) ---

@st.cache_resource
def get_db_connection():
    """Connexion unique au Google Sheet (Client Gspread)."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open("Padel_DB")
        return sheet
    except Exception as e:
        st.error(f"Erreur de connexion Google Sheets. Vérifiez vos secrets: {e}")
        st.stop()

def load_data(sheet_conn, sport_key):
    """Charge les données Joueurs et Matchs pour un sport donné."""
    config = SPORTS_CONFIG[sport_key]
    try:
        players_data = sheet_conn.worksheet(config["players_sheet"]).get_all_records()
        matches_data = sheet_conn.worksheet(config["matches_sheet"]).get_all_records()
    except gspread.WorksheetNotFound:
        st.error(f"Onglet '{config['players_sheet']}' ou '{config['matches_sheet']}' non trouvé dans votre Google Sheet.")
        st.stop()
    return players_data, matches_data

def add_match_to_db(sheet_conn, sport_key, match_dict):
    """Ajoute une ligne de match au bon onglet."""
    ws = sheet_conn.worksheet(SPORTS_CONFIG[sport_key]["matches_sheet"])
    # Créer une liste de valeurs de 1 à 12, s'assurer que les clés p1 à pN existent dans match_dict
    row_values = []
    for i in range(1, 13): # p1 à p12
        row_values.append(match_dict.get(f'p{i}', ''))
    
    row_values.extend([match_dict['score'], match_dict['winner']])
    ws.append_row([match_dict['date']] + row_values)


def delete_match_from_db(sheet_conn, sport_key, row_index):
    ws = sheet_conn.worksheet(SPORTS_CONFIG[sport_key]["matches_sheet"])
    ws.delete_rows(row_index + 2)

def add_player_to_db(sheet_conn, sport_key, name, level):
    ws = sheet_conn.worksheet(SPORTS_CONFIG[sport_key]["players_sheet"])
    ws.append_row([name, level])

def update_player_level_db(sheet_conn, sport_key, name, new_level):
    ws = sheet_conn.worksheet(SPORTS_CONFIG[sport_key]["players_sheet"])
    cell = ws.find(name)
    ws.update_cell(cell.row, 2, new_level)

def delete_player_db(sheet_conn, sport_key, name):
    ws = sheet_conn.worksheet(SPORTS_CONFIG[sport_key]["players_sheet"])
    cell = ws.find(name)
    ws.delete_rows(cell.row)

# --- LOGIQUE ELO (UNIVERSELLE) ---

def calculate_elo(rating_a, rating_b, actual_score, k=32):
    expected_score = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return rating_a + k * (actual_score - expected_score)

def process_data(players_list, matches_list):
    """Calcule le classement ELO à partir des données Joueurs et Matchs."""
    if not players_list:
        empty_df = pd.DataFrame(columns=['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D'])
        return empty_df, pd.DataFrame(), []

    players = {}
    for p in players_list:
        lvl = p.get('level', 'Intermédiaire') or 'Intermédiaire'
        players[p['name']] = {
            'level': lvl,
            'elo': STARTING_ELO.get(lvl, 1200),
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0
        }

    history_records = []
    for p_name, data in players.items():
        history_records.append({'Date': 'Début', 'Joueur': p_name, 'ELO': data['elo']})

    processed_matches = []
    
    for idx, m in enumerate(matches_list):
        # On récupère tous les joueurs d'un match jusqu'à 12
        active_players_names = [m[f'p{i}'] for i in range(1, 13) if m.get(f'p{i}')]
        
        if not active_players_names or len(active_players_names) % 2 != 0:
            continue # Ignorer les lignes sans joueurs ou avec un nombre impair
        
        total_players = len(active_players_names)
        team_size = total_players // 2
        
        t1 = active_players_names[:team_size]
        t2 = active_players_names[team_size:]
        
        # S'assurer que tous les joueurs sont enregistrés dans la BDD
        if not all(p in players for p in t1 + t2): continue
        
        m_display = m.copy()
        m_display['id'] = idx 
        processed_matches.append(m_display)

        # Calcul des ELO moyens
        avg_t1 = sum(players[p]['elo'] for p in t1) / team_size
        avg_t2 = sum(players[p]['elo'] for p in t2) / team_size
        
        res_t1 = 1 if m['winner'] == 'Team 1' else (0 if m['winner'] == 'Team 2' else 0.5)
        diff = calculate_elo(avg_t1, avg_t2, res_t1) - avg_t1
        
        # Mise à jour des stats et ELO
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
            
        for p in active_players_names:
            history_records.append({'Date': m['date'], 'Joueur': p, 'ELO': players[p]['elo']})

    df_rank = pd.DataFrame.from_dict(players, orient='index').reset_index()
    df_rank.columns = ['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D']
    df_rank['ELO'] = df_rank['ELO'].round(0).astype(int)
    
    return df_rank.sort_values(by='ELO', ascending=False), pd.DataFrame(history_records), processed_matches

# --- FONCTIONS DE RENDU UI (NOUVEAU) ---

def render_match_saisie(sheet_conn, player_list, sport_key):
    """Rend l'onglet Saisie de Match, avec taille d'équipe dynamique."""
    st.header(f"Nouveau Match {SPORTS_CONFIG[sport_key]['name']}")
    
    if sport_key == "padel":
        team_size = 2
        total_players_options = [4]
        default_total = 4
    else: # Futsal
        team_size = 4 
        total_players_options = [8, 10, 12] # 4v4, 5v5, 6v6
        default_total = st.radio("Taille du Match", total_players_options, horizontal=True)
        team_size = default_total // 2

    # Sélection des joueurs
    st.subheader(f"Sélectionnez les {default_total} joueurs (max. {team_size} par équipe)")
    
    selected_players = {}
    
    # Sélecteurs pour Team 1 (J1 à J_teamsize)
    st.markdown(f"**Team 1 ({team_size} joueurs)**")
    cols1 = st.columns(team_size)
    
    all_selected = []
    
    # Logique de sélection dynamique
    for i in range(1, team_size + 1):
        with cols1[i-1]:
            key = f"p{i}_{sport_key}"
            available = [p for p in player_list if p not in all_selected]
            selected = st.selectbox(f"J{i}", available, key=key)
            if selected and selected not in all_selected:
                all_selected.append(selected)
                selected_players[f'p{i}'] = selected
            elif not selected and player_list: # Gère le cas où la liste se vide
                 selected_players[f'p{i}'] = None


    # Sélecteurs pour Team 2 (J_teamsize+1 à J_total)
    st.markdown(f"**Team 2 ({team_size} joueurs)**")
    cols2 = st.columns(team_size)

    # Réinitialisation de la liste des disponibles pour la Team 2 (uniquement les joueurs non sélectionnés par T1)
    available_for_t2 = [p for p in player_list if p not in all_selected]
    
    for i in range(team_size + 1, default_total + 1):
        idx_in_cols = i - (team_size + 1)
        with cols2[idx_in_cols]:
            key = f"p{i}_{sport_key}"
            # Les options disponibles sont les joueurs restants + la valeur actuelle (si elle existe)
            if f'p{i}' in selected_players:
                 current_val = selected_players[f'p{i}']
            else:
                 current_val = available_for_t2[0] if available_for_t2 else None

            
            available = [p for p in player_list if p not in all_selected or p == current_val]
            
            selected = st.selectbox(f"J{i}", available, key=key)
            if selected and selected not in all_selected:
                all_selected.append(selected)
                selected_players[f'p{i}'] = selected
            elif not selected and player_list:
                selected_players[f'p{i}'] = None
    
    st.divider()
    
    # Formulaire final
    sc = st.text_input("Score (ex: 6-4 5-7 ou 10-8)", key=f"score_{sport_key}")
    da = st.date_input("Date", datetime.now(), key=f"date_{sport_key}")
    res = st.radio("Résultat", ["Team 1", "Team 2", "Egalité"], horizontal=True, key=f"res_{sport_key}")
    winner = "Team 1" if res == "Team 1" else ("Team 2" if res == "Team 2" else "Draw")
    
    # Vérification
    is_ready = len(all_selected) == default_total and sc
    
    if st.button("Valider Match", type="primary", disabled=not is_ready, key=f"btn_{sport_key}"):
        match_data = {
            'date': str(da),
            'score': sc,
            'winner': winner
        }
        # Ajouter les joueurs sélectionnés au dictionnaire de match
        for i in range(1, default_total + 1):
            match_data[f'p{i}'] = selected_players.get(f'p{i}', '')

        # S'assurer que les colonnes p(N+1) à p12 sont vides si moins de 12 joueurs
        for i in range(default_total + 1, 13):
            match_data[f'p{i}'] = ''

        add_match_to_db(sheet_conn, sport_key, match_data)
        st.success("Enregistré sur Google Sheets ! L'application va se rafraîchir.")
        st.rerun()

def render_admin_panel(sheet_conn, sport_key, raw_players, list_matches):
    """Rend l'onglet Admin (Gestion des joueurs et suppression de match)."""
    st.header(f"Gestion Admin {SPORTS_CONFIG[sport_key]['name']}")
    
    # AJOUTER JOUEUR
    with st.expander("➕ Ajouter un joueur", expanded=False):
        c_new, c_lvl = st.columns([2, 1])
        new_p = c_new.text_input("Nom", key=f"new_p_{sport_key}")
        new_lvl = c_lvl.selectbox("Niveau", list(STARTING_ELO.keys()), index=1, key=f"new_lvl_{sport_key}")
        
        if st.button("Créer Joueur", key=f"btn_create_{sport_key}"):
            if new_p and new_p not in [p['name'] for p in raw_players]:
                add_player_to_db(sheet_conn, sport_key, new_p, new_lvl)
                st.success(f"{new_p} créé !")
                st.rerun()

    # MODIFIER NIVEAU
    with st.expander("✏️ Modifier le niveau d'un joueur"):
        if raw_players:
            p_to_edit_name = st.selectbox("Choisir joueur", [p['name'] for p in raw_players], key=f"edit_player_{sport_key}")
            current_p = next((p for p in raw_players if p['name'] == p_to_edit_name), None)
            
            # Déterminer l'index actuel
            current_lvl = current_p.get('level', 'Intermédiaire') or 'Intermédiaire'
            current_lvl_idx = list(STARTING_ELO.keys()).index(current_lvl)
            
            new_lvl_edit = st.selectbox("Nouveau Niveau", list(STARTING_ELO.keys()), index=current_lvl_idx, key=f"edit_lvl_{sport_key}")
            
            if st.button("Mettre à jour le niveau", key=f"btn_update_{sport_key}"):
                update_player_level_db(sheet_conn, sport_key, p_to_edit_name, new_lvl_edit)
                st.warning(f"Niveau de {p_to_edit_name} mis à jour. Historique recalculé.")
                st.rerun()
        else:
            st.info("Aucun joueur à modifier.")

    st.divider()
    
    # SUPPRESSION MATCHS
    st.subheader("Suppression Matchs")
    if list_matches:
        # Création d'une fonction format_func pour l'affichage lisible
        def format_match(match_id):
            m = next(m for m in list_matches if m['id'] == match_id)
            players_str = ' / '.join([m[f'p{i}'] for i in range(1, 13) if m.get(f'p{i}')])
            return f"[{m['date']}] - {players_str}"

        to_del = st.selectbox("Sélectionner le match à supprimer (ID)", [m['id'] for m in list_matches][::-1], format_func=format_match, key=f"del_match_{sport_key}")
        if st.button("🗑️ Supprimer le match", key=f"btn_del_match_{sport_key}"):
            delete_match_from_db(sheet_conn, sport_key, to_del)
            st.success("Match supprimé ! Rechargement...")
            st.rerun()
    else:
        st.info("Aucun match à supprimer.")
        
    # SUPPRESSION JOUEUR (ZONE DE DANGER)
    with st.expander("☠️ Zone de danger (Supprimer Joueur)"):
        d = st.selectbox("Supprimer définitivement (Attention, retire le joueur de tous les historiques)", [p['name'] for p in raw_players], key=f"del_player_{sport_key}")
        if st.button("Supprimer Joueur", key=f"btn_del_player_{sport_key}"):
            delete_player_db(sheet_conn, sport_key, d)
            st.error(f"Joueur {d} supprimé ! Rechargement...")
            st.rerun()

# --- FONCTION PRINCIPALE DE RENDU PAR SPORT ---

def render_sport_page(sheet_conn, sport_key, IS_ADMIN):
    """Contient toute la logique d'une page (Padel ou Futsal)."""
    
    raw_players, raw_matches = load_data(sheet_conn, sport_key)
    df_rank, df_hist, list_matches = process_data(raw_players, raw_matches)
    
    player_list = [p['name'] for p in raw_players]
    
    tab_names = ["🏆 Classement", "📈 Progression", "⚖️ Générateur"]
    if IS_ADMIN: tab_names += ["➕ Saisir Match", "⚙️ Admin"]
    tabs = st.tabs(tab_names)

    # 1. CLASSEMENT
    with tabs[0]:
        st.header(f"Classement {SPORTS_CONFIG[sport_key]['name']}")
        if not df_rank.empty:
            st.dataframe(
                df_rank[['Joueur', 'Niveau', 'ELO', 'Matchs', 'V', 'N', 'D']].style.background_gradient(subset=['ELO'], cmap="Greens"), 
                use_container_width=True, 
                hide_index=True
            )
        else: st.info("Aucun joueur inscrit pour ce sport.")

    # 2. PROGRESSION
    with tabs[1]:
        st.header(f"Progression ELO {SPORTS_CONFIG[sport_key]['name']}")
        if not df_hist.empty and len(df_hist) > len(raw_players): # Plus que le point de départ
            sel = st.multiselect("Filtrer", df_rank['Joueur'].tolist(), default=df_rank['Joueur'].tolist()[:5], key=f"hist_sel_{sport_key}")
            if sel:
                sub = df_hist[df_hist['Joueur'].isin(sel)]
                y_dom = [sub['ELO'].min()-20, sub['ELO'].max()+20]
                c = alt.Chart(sub).mark_line(point=True).encode(x='Date', y=alt.Y('ELO', scale=alt.Scale(domain=y_dom)), color='Joueur', tooltip=['Date','Joueur','ELO']).interactive()
                st.altair_chart(c, use_container_width=True)
        else: st.info("Pas assez de matchs pour l'historique.")

    # 3. GENERATEUR
    with tabs[2]:
        st.header(f"Générateur d'Équipes {SPORTS_CONFIG[sport_key]['name']}")
        
        # Le générateur utilise toujours la taille minimale (4vs4=8 joueurs)
        if sport_key == "padel":
            min_players_needed = 4
        else:
            min_players_needed = 8 
            
        present = st.multiselect(f"Joueurs présents ({min_players_needed} minimum)", df_rank['Joueur'].tolist(), key=f"gen_sel_{sport_key}")
        
        if len(present) >= min_players_needed and len(present) % 2 == 0:
            total = len(present)
            team_size = total // 2
            st.subheader(f"Trouver l'équilibre pour {team_size} vs {team_size}")
            
            elos = {n: df_rank[df_rank['Joueur'] == n]['ELO'].values[0] for n in present}
            p = present
            
            # Générer toutes les combinaisons possibles (très gourmand au-delà de 10 joueurs)
            # On utilise une logique simplifiée pour trouver les meilleures paires
            import itertools
            best_diff = float('inf')
            best_matchup = []
            
            # Trouver les combinaisons de division en deux équipes de taille égale
            # C'est la partie la plus complexe en code
            for combination in itertools.combinations(p, team_size):
                team1 = list(combination)
                team2 = [player for player in p if player not in team1]
                
                # Vérifier pour éviter la duplication (ex: A/B vs C/D et C/D vs A/B)
                if team1[0] > team2[0]: 
                    continue 
                
                avg_elo_t1 = sum(elos[j] for j in team1) / team_size
                avg_elo_t2 = sum(elos[j] for j in team2) / team_size
                
                diff = abs(avg_elo_t1 - avg_elo_t2)
                
                if diff < best_diff:
                    best_diff = diff
                    best_matchup = [{'Txt': f"Team 1 ({round(avg_elo_t1)}) : {' / '.join(team1)}", 'Txt2': f"Team 2 ({round(avg_elo_t2)}) : {' / '.join(team2)}", 'Diff': diff}]
                elif diff == best_diff:
                     best_matchup.append({'Txt': f"Team 1 ({round(avg_elo_t1)}) : {' / '.join(team1)}", 'Txt2': f"Team 2 ({round(avg_elo_t2)}) : {' / '.join(team2)}", 'Diff': diff})
                     
            if best_matchup:
                st.success(f"Meilleure option (Écart ELO: {best_diff:.1f} points):")
                st.markdown(f"**{best_matchup[0]['Txt']}**")
                st.markdown(f"**{best_matchup[0]['Txt2']}**")
            
        else:
            st.warning(f"Sélectionnez un nombre pair de joueurs (minimum {min_players_needed}).")


    # 4. SAISIE (ADMIN)
    if IS_ADMIN and "➕ Saisir Match" in tab_names:
        with tabs[3]:
            if len(player_list) < (SPORTS_CONFIG[sport_key]['team_size'] * 2):
                st.warning(f"Besoin d'au moins {SPORTS_CONFIG[sport_key]['team_size'] * 2} joueurs pour saisir un match {SPORTS_CONFIG[sport_key]['name']}.")
            else:
                render_match_saisie(sheet_conn, player_list, sport_key)

    # 5. ADMIN (ADMIN)
    if IS_ADMIN and "⚙️ Admin" in tab_names:
        with tabs[4]:
            render_admin_panel(sheet_conn, sport_key, raw_players, list_matches)

# --- FLUX PRINCIPAL DE L'APPLICATION ---

def main():
    st.title("🏆 Sport ELO Manager")

    # 1. CONNEXION BDD
    sheet_conn = get_db_connection()
    
    # 2. GESTION DES DROITS
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2643/2643479.png", width=50)
        
        # SÉLECTION DU SPORT (MENU PRINCIPAL)
        sport_choice = st.radio(
            "Choisissez le sport :",
            options=["padel", "futsal"],
            format_func=lambda x: SPORTS_CONFIG[x]['name']
        )
        st.divider()
        
        # LOGIN ADMIN
        st.write("## Espace Admin")
        pwd = st.text_input("Mot de passe Admin", type="password")
        IS_ADMIN = (pwd == st.secrets.get("ADMIN_PASSWORD", "padel2024"))
        if IS_ADMIN: st.success("Mode Admin activé ✅")
        else: st.info("Mode Visiteur (Lecture seule)")

    st.header(SPORTS_CONFIG[sport_choice]['name'])
    
    # 3. RENDU DE LA PAGE DU SPORT SÉLECTIONNÉ
    render_sport_page(sheet_conn, sport_choice, IS_ADMIN)

if __name__ == "__main__":
    main()
