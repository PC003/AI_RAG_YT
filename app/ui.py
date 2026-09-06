"""Streamlit Frontend for YouTube Playlist RAG."""

import streamlit as st
import os
import sys

# Ensure the app module can be found when running via `streamlit run app/ui.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.processor import process_playlist
from app.rag.retriever import answer_query
from app.models.schemas import IngestionStats


st.set_page_config(
    page_title="YouTube Playlist RAG",
    page_icon="▶️",
    layout="wide"
)

st.title("YouTube Playlist RAG 📚")
st.markdown("Chat with the contents of a YouTube Playlist using Local Whisper and Embeddings.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Ingest Playlist"])

# --- Tab 1: Chat ---
with tab1:
    st.header("Ask a Question")

    query = st.text_input("What do you want to know?", placeholder="e.g., What is the difference between LL and LR parsing?")

    if st.button("Ask", type="primary"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    response = answer_query(query)

                    st.markdown("### Answer")
                    st.info(response.answer)

                    if response.sources:
                        st.markdown("### Sources")
                        # Display sources cleanly
                        for i, source in enumerate(response.sources, 1):
                            with st.expander(f"📍 {source.video_title} ({source.timestamp_display})"):
                                st.markdown(f"**Video:** {source.video_title}")
                                st.markdown(f"**Timestamp:** {source.timestamp_display}")
                                st.markdown(f"[▶️ Watch from this point]({source.youtube_url})")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")


# --- Tab 2: Ingestion ---
with tab2:
    st.header("Process a new Playlist")

    playlist_url = st.text_input("YouTube Playlist URL")

    if st.button("Process Playlist", type="primary"):
        if not playlist_url:
            st.warning("Please enter a YouTube Playlist URL.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg: str, current: int, total: int):
                if total > 0:
                    progress = min(current / total, 1.0)
                    progress_bar.progress(progress)
                status_text.text(f"[{current}/{total}] {msg}")

            with st.spinner("Initializing ingestion (this may take a while)..."):
                try:
                    stats: IngestionStats = process_playlist(playlist_url, update_progress)

                    st.success("✅ Ingestion Complete!")

                    cols = st.columns(4)
                    cols[0].metric("Videos Found", stats.videos_found)
                    cols[1].metric("Videos Indexed", stats.videos_processed)
                    cols[2].metric("Videos Skipped", stats.videos_skipped)
                    cols[3].metric("Vectors Created", stats.chunks_created)

                    if stats.videos_failed > 0:
                        st.error(f"Failed to process {stats.videos_failed} videos.")
                        with st.expander("View failures"):
                            for f in stats.failed_videos:
                                st.write(f)

                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
