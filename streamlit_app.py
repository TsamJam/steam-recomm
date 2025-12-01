import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= KONFIGURASI =================
# Coba baca dari Secrets (untuk Deploy), jika gagal pakai Key Manual (untuk Lokal)
try:
    STEAM_API_KEY = st.secrets["STEAM_API_KEY"]


CSV_SOURCE = 'steam_large_dataset.csv'
JSONL_DATA = 'steam_data.jsonl'
MATRIX_FILE = 'steam_similarity_matrix.parquet'

st.set_page_config(
    page_title="Steam Game Scout",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CLASS LOGIKA REKOMENDASI =================
class SteamRecommender:
    def __init__(self):
        self.similarity_df = None
        self.available_games = []
        self.game_id_map = {} 
        self.analytics_cache = {} 
        
        # Setup Session agar koneksi stabil
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _ensure_data_exists(self):
        """Memastikan data JSONL tersedia, jika tidak buat dari CSV."""
        if not os.path.exists(JSONL_DATA):
            if os.path.exists(CSV_SOURCE):
                with st.status("⚙️ Mengonversi CSV ke JSONL...", expanded=True) as status:
                    try:
                        df_csv = pd.read_csv(CSV_SOURCE, dtype={'user_id': str})
                        df_csv.to_json(JSONL_DATA, orient='records', lines=True, force_ascii=False)
                        status.update(label="✅ Konversi Selesai!", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"❌ Gagal konversi: {e}")
                        return False
        return True

    def load_model(self):
        """Memuat atau Melatih Model Collaborative Filtering."""
        self._ensure_data_exists()

        if not os.path.exists(JSONL_DATA):
             st.error(f"❌ File '{JSONL_DATA}' tidak ditemukan! Pastikan file CSV diupload.")
             return False

        # 1. Load Mapping ID (Nama -> ID) untuk Analytics
        try:
            df = pd.read_json(JSONL_DATA, lines=True)
            temp_map = df.drop_duplicates(subset=['game_name'])
            self.game_id_map = dict(zip(temp_map.game_name, temp_map.game_id))
        except Exception as e:
            st.error(f"❌ Gagal membaca data: {e}")
            return False

        # 2. Cek Cache Parquet (Agar Cepat)
        if os.path.exists(MATRIX_FILE):
            try:
                self.similarity_df = pd.read_parquet(MATRIX_FILE)
                self.available_games = self.similarity_df.index.tolist()
                return True
            except Exception as e:
                st.warning(f"⚠️ Cache rusak, membangun ulang model...")

        # 3. Build Baru (Jika cache tidak ada)
        with st.spinner("⚙️ Sedang melatih model AI dari data user... (Hanya sekali)"):
            game_counts = df['game_name'].value_counts()
            # Filter game yang minimal dimainkan 5 user agar akurat
            popular_games = game_counts[game_counts >= 5].index
            df_filtered = df[df['game_name'].isin(popular_games)]
            
            # Logarithmic Scaling untuk Playtime
            df_filtered['log_playtime'] = np.log1p(df_filtered['playtime_forever'])
            
            # Pivot Table (User x Game)
            pivot_table = df_filtered.pivot_table(
                index='game_name', 
                columns='user_id', 
                values='log_playtime', 
                fill_value=0
            )
            
            # Hitung Cosine Similarity (Matematika Berat)
            sparse_pivot = csr_matrix(pivot_table.values)
            similarity_matrix = cosine_similarity(sparse_pivot)
            
            # Simpan ke DataFrame & File
            self.similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)
            self.similarity_df.to_parquet(MATRIX_FILE)
            self.available_games = self.similarity_df.index.tolist()
        
        return True

    def fetch_game_analytics(self, game_name):
        """Mengambil data live dari Steam (Harga, Gambar, dll)."""
        if game_name in self.analytics_cache:
            return self.analytics_cache[game_name]

        app_id = self.game_id_map.get(game_name)
        
        # Default Data
        analytics = {
            "name": game_name,
            "app_id": int(app_id) if app_id else None,
            "image": "https://via.placeholder.com/460x215?text=No+Image",
            "price": "N/A",
            "description": "Deskripsi tidak tersedia.",
            "genres": [],
            "current_players": 0,
            "review_sentiment": "Unknown",
            "store_link": f"https://store.steampowered.com/app/{app_id}/" if app_id else "#"
        }

        if not app_id: return analytics

        try:
            # 1. Store Data (Harga, Gambar)
            store_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=id"
            store_resp = self.session.get(store_url, timeout=2).json()
            
            if str(app_id) in store_resp and store_resp[str(app_id)]['success']:
                data = store_resp[str(app_id)]['data']
                analytics['image'] = data.get('header_image', analytics['image'])
                analytics['description'] = data.get('short_description', analytics['description'])
                analytics['genres'] = [g['description'] for g in data.get('genres', [])[:3]]
                
                if data.get('is_free'):
                    analytics['price'] = "Free to Play"
                elif 'price_overview' in data:
                    analytics['price'] = data['price_overview']['final_formatted']

            # 2. Current Players
            player_url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={app_id}"
            player_resp = self.session.get(player_url, timeout=2).json()
            if player_resp['response']['result'] == 1:
                analytics['current_players'] = player_resp['response']['player_count']

            # 3. Reviews
            review_url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all&purchase_type=all"
            review_resp = self.session.get(review_url, timeout=2).json()
            if review_resp['success']:
                analytics['review_sentiment'] = review_resp['query_summary'].get('review_score_desc', 'Unknown')

        except: pass

        self.analytics_cache[game_name] = analytics
        return analytics

    def get_user_library(self, steam_id):
        """Mengambil game user dari API Steam."""
        url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
        params = {'key': STEAM_API_KEY, 'steamid': steam_id, 'include_appinfo': 1, 'format': 'json'}
        try:
            resp = self.session.get(url, params=params, timeout=10).json()
            if 'response' in resp and 'games' in resp['response']:
                return resp['response']['games']
        except: pass
        return []

    def recommend_weighted(self, user_games_dict, exclude_games, top_n=12):
        """Menghitung rekomendasi dengan bobot playtime."""
        if self.similarity_df is None: return []
        
        # Hanya hitung game yang ada di database kita
        valid_games = {g: t for g, t in user_games_dict.items() if g in self.available_games}
        if not valid_games: return []

        # Hitung Skor: Vektor Similarity * Log(Playtime)
        total_scores = pd.Series(0.0, index=self.similarity_df.index)
        for game_name, playtime in valid_games.items():
            weight = np.log1p(playtime)
            total_scores += (self.similarity_df[game_name] * weight)

        # Buang game yang sudah dimiliki
        games_to_drop = set(user_games_dict.keys())
        if exclude_games: games_to_drop.update(exclude_games)
        
        total_scores = total_scores.drop(index=list(games_to_drop), errors='ignore')
        recommended = total_scores.sort_values(ascending=False).head(top_n)

        # Ambil detail (Analytics)
        results = []
        for game, score in recommended.items():
            game_data = self.fetch_game_analytics(game)
            game_data['similarity_score'] = round(score, 2)
            results.append(game_data)
            
        return results

# ================= FUNGSI UTAMA (CACHE) =================
@st.cache_resource
def get_recommender():
    rec = SteamRecommender()
    rec.load_model()
    return rec

# ================= TAMPILAN UI =================
def main():
    # Sidebar untuk Input
    st.sidebar.title("🎮 Steam Game Scout")
    st.sidebar.write("Temukan *Hidden Gems* berdasarkan riwayat bermainmu!")
    
    steam_id_input = st.sidebar.text_input("Masukkan Steam ID:", placeholder="Contoh: 76561198998178100")
    st.sidebar.caption("Pastikan profil Steam diset ke PUBLIC agar bisa dibaca.")
    
    analyze_btn = st.sidebar.button("🔍 Analisis & Cari Game", type="primary")
    
    st.title("Rekomendasi Game Cerdas")
    st.write("Sistem ini menggunakan **Collaborative Filtering** dengan pembobotan waktu bermain untuk menemukan game yang benar-benar sesuai selera Anda.")
    
    # Inisialisasi Model
    recommender = get_recommender()

    if analyze_btn and steam_id_input:
        if not steam_id_input.isdigit() or len(steam_id_input) != 17:
            st.error("⚠️ Format Steam ID salah! Harus 17 digit angka.")
            return

        with st.status("🤖 Sedang bekerja...", expanded=True) as status:
            st.write("📡 Mengambil library game dari Steam...")
            owned_games = recommender.get_user_library(steam_id_input)
            
            if not owned_games:
                status.update(label="❌ Gagal!", state="error")
                st.error("Tidak dapat mengambil data. Pastikan Profil Public atau ID benar.")
                return

            all_owned_names = [g['name'] for g in owned_games]
            
            st.write("📊 Menganalisis pola bermain (Top 15 Game)...")
            sorted_games = sorted(owned_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
            top_games = sorted_games[:15]
            user_games_map = {g['name']: g.get('playtime_forever', 0) for g in top_games}

            st.write("🧠 Menghitung kecocokan matriks...")
            recommendations = recommender.recommend_weighted(
                user_games_dict=user_games_map,
                exclude_games=all_owned_names,
                top_n=12
            )
            status.update(label="✅ Selesai!", state="complete", expanded=False)

        # Tampilkan Hasil
        if not recommendations:
            st.warning("Tidak ada rekomendasi yang cukup kuat ditemukan. Coba mainkan lebih banyak game variatif!")
        else:
            st.success(f"Ditemukan rekomendasi berdasarkan {len(top_games)} game favoritmu!")
            
            # Grid Layout Custom
            cols = st.columns(3)
            for idx, game in enumerate(recommendations):
                with cols[idx % 3]:
                    with st.container(border=True):
                        # Gambar Header
                        if game['image']:
                            st.image(game['image'], use_container_width=True)
                        
                        st.subheader(game['name'])
                        
                        # Info Baris 1
                        c1, c2 = st.columns([1, 1])
                        c1.caption("Score Kecocokan")
                        c1.write(f"⭐ **{game['similarity_score']}**")
                        
                        c2.caption("Status Review")
                        sentiment = game.get('review_sentiment', 'Unknown')
                        color = "green" if "Positive" in sentiment else "orange" if "Mixed" in sentiment else "grey"
                        c2.markdown(f":{color}[{sentiment}]")

                        # Info Baris 2
                        c3, c4 = st.columns([1, 1])
                        c3.caption("Harga")
                        c3.write(f"💰 {game.get('price', 'N/A')}")
                        
                        c4.caption("Pemain Aktif")
                        c4.write(f"👥 {game.get('current_players', 0):,}")

                        # Tombol Link
                        st.link_button("Lihat di Steam", game['store_link'], use_container_width=True)

    elif analyze_btn:
        st.sidebar.error("Mohon isi Steam ID terlebih dahulu!")

    st.divider()
    st.caption("Dibuat dengan Python & Streamlit | Data provided by Valve Corporation")

if __name__ == "__main__":
    main()