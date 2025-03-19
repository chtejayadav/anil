import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    file_path = "anil2.xlsx"  # Ensure the file is in the same directory
    df = pd.read_excel(file_path)
    df = df[['title', 'year', 'genre', 'rating', 'desc']].dropna()
    df['year'] = df['year'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df

df = load_data()

st.title("🎬 Movie Recommendation System")

# Filters
selected_genre = st.selectbox("Select Genre", sorted(set(
    [g.strip() for sublist in df['genre'].dropna().str.split(',') for g in sublist])
))

selected_year = st.selectbox("Select Year", sorted(df['year'].unique(), reverse=True))

min_rating = st.slider("Minimum Rating", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

# Filter movies
filtered_movies = df[(df['genre'].str.contains(selected_genre, case=False, na=False)) &
                     (df['year'] == selected_year) &
                     (df['rating'] >= min_rating)]

# Display results
st.subheader("Recommended Movies")
if not filtered_movies.empty:
    for _, row in filtered_movies.iterrows():
        st.markdown(f"**{row['title']} ({row['year']})**")
        st.text(f"⭐ {row['rating']}")
        st.write(row['desc'])
        st.markdown("---")
else:
    st.write("No movies found. Try adjusting filters!")
