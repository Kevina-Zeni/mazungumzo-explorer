import streamlit as st
from langchain_groq import ChatGroq
from utils import (
    load_manifest, load_turns, load_topics, load_keywords,
    load_vector_store, load_bm25, hybrid_retrieve
)

st.set_page_config(page_title="Mazungumzo Explorer", layout="wide")

manifest = load_manifest()
episode_options = {f"{ep['episode_id']} — {ep['filename']} ({ep['format']})": ep for ep in manifest}

st.sidebar.title("Mazungumzo Pipeline")
selected_label = st.sidebar.selectbox("Select an episode", list(episode_options.keys()))
selected_ep = episode_options[selected_label]
ep_id = selected_ep["episode_id"]

st.title(selected_ep["filename"])
st.caption(f"{selected_ep['format'].capitalize()} · Guests: {selected_ep['guests']}")

tab_transcript, tab_themes, tab_query = st.tabs(["Transcript", "Themes", "Query"])

with tab_transcript:
    turns = load_turns(ep_id)
    for turn in turns:
        ts = turn["start"]
        ts_str = f"{int(ts // 60)}m {int(ts % 60)}s" if ts is not None else ""
        st.markdown(f"**{turn['speaker']}** _{ts_str}_")
        st.write(turn["text"])
        st.divider()

with tab_themes:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Themes in This Episode")
        st.bar_chart(load_topics(ep_id))
    with col2:
        st.subheader("Top Keywords in This Episode")
        st.bar_chart(load_keywords(ep_id))

with tab_query:
    st.subheader("Ask a question about this episode")
    question = st.text_input("e.g. What challenges were discussed in relation to open science?")
    if question:
        with st.spinner("Retrieving and generating answer..."):
            store, _ = load_vector_store(ep_id)
            bm25_data = load_bm25(ep_id)
            contexts = hybrid_retrieve(question, store, bm25_data, k=3)
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
            prompt = f"""Answer using ONLY the context below. If insufficient, say "Insufficient context provided."

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""
            answer = llm.invoke(prompt).content
        st.markdown("### Answer")
        st.write(answer)
        with st.expander("View retrieved source passages"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Passage {i}**")
                st.write(c)
