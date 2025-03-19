import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# ✅ Ensure set_page_config() is the first Streamlit command
st.set_page_config(page_title="MOVIES RECOMMENDATION SYSTEM", layout="wide")

# Function to set background image and change text color
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set background image
image_path = "Cinema Projector intro.gif"  # Ensure the image exists in the same directory
set_background(image_path)

# Load dataset
@st.cache_data
def load_data():
    file_path = (r"IMDB_10000.csv")  # Ensure this file is in the same directory
    df = pd.read_csv(file_path)
    
    # Data Cleaning
    df['runtime'] = df['runtime'].astype(str).str.extract('(\d+)').astype(float)
    df['votes'] = df['votes'].astype(str).str.replace(',', '').astype(float)
    df[['desc', 'genre']] = df[['desc', 'genre']].fillna('')
    
    # Feature Engineering
    df['features'] = df['genre'] + " " + df['desc']

    return df

movies_df = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['features'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def get_recommendations(title):
    if title not in movies_df['title'].values:
        return []

    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['title', 'genre']]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

# Select movie from dropdown
selected_movie = st.selectbox("Select a movie:", movies_df['title'].values)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie)
    
    if recommendations.empty:
        st.write("❌ No recommendations found.")
    else:
        st.write("### 🔥 Top 5 Recommendations:")
        st.dataframe(recommendations)
