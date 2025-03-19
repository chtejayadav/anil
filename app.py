import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    file_path = "anil.csv"  # Ensure the file is in the same directory
    df = pd.read_csv(file_path)
    df = df[['title', 'genre', 'desc']].dropna()
    return df

df = load_data()

# Create a similarity matrix based on descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['desc'])
similarity_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(movie_name):
    if movie_name not in df['title'].values:
        return []
    idx = df[df['title'] == movie_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genre', 'desc']]

st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox("Select a movie", df['title'].unique())

if movie_name:
    recommendations = get_recommendations(movie_name)
    st.subheader("Recommended Movies")
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}**")
            st.text(f"🎭 {row['genre']}")
            st.write(row['desc'])
            st.markdown("---")
    else:
        st.write("No recommendations found. Try a different movie!")
