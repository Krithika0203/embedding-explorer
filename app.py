import streamlit as st
import pandas as pd
from embeddings import generate_embedding, cosine_similarity
from data_store import DataStore

st.set_page_config(
    page_title="Embedding Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Embedding Explorer")
st.caption("Generate embeddings and perform semantic search")

if "store" not in st.session_state:
    st.session_state.store = DataStore()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("📥 Add Text")

    new_text = st.text_area("Enter text")

    if st.button("Generate & Store Embedding"):

        st.session_state.store.add_text(new_text)

        emb = generate_embedding(new_text)

        st.success("Embedding generated!")

        st.write("Embedding length:", len(emb))

        st.write("First 10 values:", emb[:10])

with col2:

    st.subheader("🔎 Semantic Search")

    query = st.text_input("Search text")

    if st.button("Find Similar Text"):

        texts, embeddings = st.session_state.store.get_all()

        if len(texts) == 0:

            st.warning("No stored texts yet.")

        else:

            query_emb = generate_embedding(query)

            scores = []

            for emb in embeddings:

                score = cosine_similarity(query_emb, emb)

                scores.append(score)

            result_df = pd.DataFrame({
                "Text": texts,
                "Similarity": scores
            })

            result_df = result_df.sort_values(
                by="Similarity",
                ascending=False
            )

            st.dataframe(result_df)

st.markdown("---")

st.subheader("📊 Stored Data")

texts, embeddings = st.session_state.store.get_all()

st.write("Total stored texts:", len(texts))

if len(texts) > 0:

    df = pd.DataFrame({"Stored Text": texts})

    st.dataframe(df)