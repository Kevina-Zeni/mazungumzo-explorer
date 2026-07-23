import streamlit as st
import pandas as pd
import re
import json
import plotly.graph_objects as go
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

tab_transcript, tab_themes, tab_query, tab_audio = st.tabs(
    ["Transcript", "Themes", "Query", "Analyze Your Audio"]
)

with tab_transcript:
    turns = load_turns(ep_id)

    df = pd.DataFrame(turns)
    df["duration"] = df["end"] - df["start"]
    df["start_min"] = df["start"] / 60.0
    df["end_min"] = df["end"] / 60.0
    df["duration_min"] = df["end_min"] - df["start_min"]

    st.subheader("Speaker Timeline")

    speakers_order = df["speaker"].unique().tolist()
    st.caption(f"{len(speakers_order)} speaker(s) detected in this episode")
    fig = go.Figure()

    for speaker in speakers_order:
        speaker_df = df[df["speaker"] == speaker]
        fig.add_trace(go.Bar(
            y=[speaker] * len(speaker_df),
            x=speaker_df["duration_min"],
            base=speaker_df["start_min"],
            orientation="h",
            name=speaker,
            hovertext=speaker_df["text"],
            hoverinfo="text",
        ))

    fig.update_layout(
        xaxis_title="Episode Timeline (Minutes)",
        yaxis=dict(autorange="reversed", title=None),
        showlegend=False,
        height=150 + len(speakers_order) * 60,
    )
    st.plotly_chart(fig, use_container_width=True)

    talk_pct = (df.groupby("speaker")["duration"].sum() / df["duration"].sum() * 100).round(1)
    first_appearance = df.groupby("speaker")["start"].min()

    cols = st.columns(len(talk_pct))
    for col, (speaker, pct) in zip(cols, talk_pct.items()):
        ts = first_appearance[speaker]
        col.metric(speaker, f"{pct}%", f"first at {int(ts//60)}m {int(ts%60)}s")

    st.divider()

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
        keywords_data = load_keywords(ep_id)
        keywords_df = pd.DataFrame(keywords_data, columns=["word", "count"]).set_index("word")
        st.bar_chart(keywords_df)

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

with tab_audio:
    st.subheader("Analyze Your Audio")
    st.caption("Upload a clip under 90 seconds.")

    MAX_CLIP_SECONDS = 90

    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], key="demo_audio_upload")

    if audio_file is not None:
        import tempfile
        import os
        import soundfile as sf

        suffix = os.path.splitext(audio_file.name)[1]
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())

        try:
            info = sf.info(tmp_path)
            duration = info.frames / info.samplerate
        except Exception as e:
            st.error(f"Couldn't read this audio file: {e}")
            duration = None

        if duration is not None and duration > MAX_CLIP_SECONDS:
            st.error(f"This clip is {duration:.0f}s long — please upload something under {MAX_CLIP_SECONDS}s for this demo.")
        elif duration is not None:
            st.success(f"Clip loaded: {duration:.1f}s. Ready to process.")
            st.session_state["demo_audio_path"] = tmp_path

            if st.button("Transcribe this clip"):
                with st.spinner("Transcribing..."):
                    from groq import Groq
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

                    with open(tmp_path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=f,
                            model="whisper-large-v3-turbo",
                            response_format="verbose_json",
                        )

                    st.session_state["demo_transcript_segments"] = transcription.segments
                    st.session_state["demo_language"] = transcription.language
                    st.success("Transcription complete.")

            if "demo_transcript_segments" in st.session_state:
                st.caption(f"Detected language: {st.session_state['demo_language']}")

            if "demo_transcript_segments" in st.session_state:
                if st.button("Identify speakers"):
                    with st.spinner("Identifying speakers (slower step, runs on CPU)..."):
                        from pyannote.audio import Pipeline

                        @st.cache_resource
                        def load_diarization_pipeline():
                            return Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                token=st.secrets["HF_TOKEN"],
                            )

                        diarization_pipeline = load_diarization_pipeline()
                        diarization = diarization_pipeline(tmp_path)

                        speaker_segments = []
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            speaker_segments.append({
                                "speaker": speaker,
                                "start": turn.start,
                                "end": turn.end,
                            })

                        st.session_state["demo_speaker_segments"] = speaker_segments
                        st.success(f"Found {len(set(s['speaker'] for s in speaker_segments))} speaker(s).")

            if "demo_speaker_segments" in st.session_state and "demo_transcript_segments" in st.session_state:
                if "demo_merged_turns" not in st.session_state:
                    merged = []
                    for t_seg in st.session_state["demo_transcript_segments"]:
                        mid = (t_seg["start"] + t_seg["end"]) / 2
                        speaker = "UNKNOWN"
                        for s_seg in st.session_state["demo_speaker_segments"]:
                            if s_seg["start"] <= mid <= s_seg["end"]:
                                speaker = s_seg["speaker"]
                                break
                        merged.append({
                            "speaker": speaker,
                            "start": t_seg["start"],
                            "end": t_seg["end"],
                            "text": t_seg["text"].strip(),
                        })
                    st.session_state["demo_merged_turns"] = merged

                if "demo_named_turns" not in st.session_state:
                    INTRO_PATTERNS = [
                        r"\bmy name is ([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+){0,2})",
                        r"\bi'?m ([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+){0,2})",
                        r"\bi am ([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+){0,2})",
                        r"\bthis is ([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+){0,2})\s+speaking",
                    ]
                    INTRO_WINDOW_SECONDS = 60

                    candidates = {}
                    for seg in st.session_state["demo_merged_turns"]:
                        if seg["start"] > INTRO_WINDOW_SECONDS or seg["speaker"] in candidates:
                            continue
                        for pattern in INTRO_PATTERNS:
                            match = re.search(pattern, seg["text"])
                            if match:
                                name = match.group(1).strip()
                                if 1 <= len(name.split()) <= 3:
                                    candidates[seg["speaker"]] = name
                                break

                    named_turns = []
                    for seg in st.session_state["demo_merged_turns"]:
                        label = candidates.get(seg["speaker"], seg["speaker"].replace("_", " ").title())
                        named_turns.append({**seg, "speaker_label": label})
                    st.session_state["demo_named_turns"] = named_turns
                    st.session_state["demo_detected_names"] = candidates

                if st.session_state.get("demo_detected_names"):
                    st.write("Detected names:", st.session_state["demo_detected_names"])

                named_turns = st.session_state["demo_named_turns"]

                demo_df = pd.DataFrame(named_turns)
                demo_df["duration"] = demo_df["end"] - demo_df["start"]
                demo_df["start_min"] = demo_df["start"] / 60.0
                demo_df["duration_min"] = demo_df["duration"] / 60.0

                demo_speakers = demo_df["speaker_label"].unique().tolist()
                st.subheader("Speaker Timeline")
                st.caption(f"{len(demo_speakers)} speaker(s) detected in this clip")

                demo_fig = go.Figure()
                for speaker in demo_speakers:
                    spk_df = demo_df[demo_df["speaker_label"] == speaker]
                    demo_fig.add_trace(go.Bar(
                        y=[speaker] * len(spk_df),
                        x=spk_df["duration_min"],
                        base=spk_df["start_min"],
                        orientation="h",
                        name=speaker,
                        hovertext=spk_df["text"],
                        hoverinfo="text",
                    ))
                demo_fig.update_layout(
                    xaxis_title="Clip Timeline (Minutes)",
                    yaxis=dict(autorange="reversed", title=None),
                    showlegend=False,
                    height=150 + len(demo_speakers) * 60,
                )
                st.plotly_chart(demo_fig, use_container_width=True)

                talk_pct = (demo_df.groupby("speaker_label")["duration"].sum() / demo_df["duration"].sum() * 100).round(1)
                first_appearance = demo_df.groupby("speaker_label")["start"].min()
                demo_cols = st.columns(len(talk_pct))
                for col, (speaker, pct) in zip(demo_cols, talk_pct.items()):
                    ts = first_appearance[speaker]
                    col.metric(speaker, f"{pct}%", f"first at {int(ts//60)}m {int(ts%60)}s")

                with st.expander("Full transcript"):
                    for seg in named_turns:
                        st.markdown(f"**{seg['speaker_label']}** [{seg['start']:.1f}s–{seg['end']:.1f}s]")
                        st.write(seg["text"])
                        st.divider()

                full_text = " ".join(seg["text"] for seg in named_turns)

                if st.button("Extract themes"):
                    with st.spinner("Extracting themes..."):
                        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

                        theme_prompt = f"""
You are analyzing a scholarly interview transcript.
Identify the major themes discussed in the transcript.
Return ONLY valid JSON in this format:
[
  {{
    "theme": "Theme name",
    "description": "Short explanation of the theme",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "importance_score": 1
  }}
]
Use importance_score from 1 to 5, where:
1 = minor theme
5 = dominant theme

Transcript:
{full_text[:12000]}
"""
                        theme_response = llm.invoke(theme_prompt)
                        theme_text = theme_response.content.strip()
                        json_match = re.search(r"\[.*\]", theme_text, re.DOTALL)

                        if json_match:
                            themes_json = json.loads(json_match.group(0))
                            themes_df = pd.DataFrame(themes_json)
                            st.session_state["demo_themes_df"] = themes_df

                            relationship_prompt = f"""
You are analyzing a scholarly interview.
Given the following themes:
{themes_df['theme'].tolist()}

Identify meaningful relationships between themes.
Return ONLY valid JSON as a list of edges in this format:
[
  {{"source": "Theme A", "target": "Theme B", "relationship": "short description"}}
]
Only include relationships that are clearly meaningful.
"""
                            rel_response = llm.invoke(relationship_prompt)
                            rel_text = rel_response.content.strip()
                            rel_match = re.search(r"\[.*\]", rel_text, re.DOTALL)
                            relationships = json.loads(rel_match.group(0)) if rel_match else []
                            st.session_state["demo_relationships"] = relationships
                        else:
                            st.error("Couldn't extract themes from this clip — try a longer or more topic-dense recording.")

                if "demo_themes_df" in st.session_state:
                    import networkx as nx

                    st.subheader("Theme Knowledge Graph")
                    st.caption("LLM-extracted themes and relationships for this clip "
                               "(display-layer method — distinct from the thesis's NMF topic model).")

                    themes_df = st.session_state["demo_themes_df"]
                    relationships = st.session_state.get("demo_relationships", [])

                    G = nx.DiGraph()
                    for _, row in themes_df.iterrows():
                        G.add_node(row["theme"], importance=row["importance_score"], description=row["description"])
                    for rel in relationships:
                        if rel["source"] in G.nodes and rel["target"] in G.nodes:
                            G.add_edge(rel["source"], rel["target"], relationship=rel.get("relationship", ""))

                    pos = nx.spring_layout(G, seed=42)
                    edge_x, edge_y, annotations = [], [], []
                    for u, v in G.edges():
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
                        annotations.append(dict(
                            ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, opacity=0.6,
                        ))
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                             line=dict(width=1, color="#999"), hoverinfo="none")

                    node_x, node_y, node_text, node_size, hover_text = [], [], [], [], []
                    for node, data in G.nodes(data=True):
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                        node_size.append(15 + data.get("importance", 1) * 6)
                        hover_text.append(f"{node}<br>{data.get('description', '')}")

                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode="markers+text",
                        text=node_text, textposition="top center",
                        hovertext=hover_text, hoverinfo="text",
                        marker=dict(size=node_size, color="#2166ac"),
                    )
                    graph_fig = go.Figure(data=[edge_trace, node_trace])
                    graph_fig.update_layout(
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=20, b=0), annotations=annotations,
                    )
                    st.plotly_chart(graph_fig, use_container_width=True)

                    with st.expander("Relationship details"):
                        for rel in relationships:
                            st.write(f"**{rel['source']} → {rel['target']}**: {rel.get('relationship', '')}")

                st.subheader("Ask a question about this clip")
                demo_question = st.text_input("Your question", key="demo_query_input")

                if demo_question:
                    with st.spinner("Retrieving and answering..."):
                        from langchain_huggingface import HuggingFaceEmbeddings
                        from langchain_chroma import Chroma
                        from langchain_core.documents import Document

                        @st.cache_resource
                        def load_demo_embedding_model():
                            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

                        embeddings = load_demo_embedding_model()
                        docs = [
                            Document(page_content=seg["text"], metadata={"speaker": seg["speaker_label"]})
                            for seg in named_turns
                        ]
                        demo_store = Chroma.from_documents(
                            docs, embeddings,
                            collection_metadata={"hnsw:space": "cosine"},
                        )
                        results = demo_store.similarity_search(demo_question, k=3)

                        context_text = "\n".join(r.page_content for r in results)
                        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
                        demo_prompt = f"""Answer using ONLY the context below. If insufficient, say "Insufficient context provided."

Context:
{context_text}

Question: {demo_question}
Answer:"""
                        demo_answer = llm.invoke(demo_prompt).content

                    st.markdown("### Answer")
                    st.write(demo_answer)
                    with st.expander("View retrieved passages"):
                        for r in results:
                            st.write(f"[{r.metadata['speaker']}] {r.page_content}")
