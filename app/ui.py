"""Streamlit frontend for YouTube Playlist RAG."""

import os
import sys

import streamlit as st

# Ensure the app module can be found when running via `streamlit run app/ui.py`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.processor import process_playlist
from app.models.schemas import IngestionStats
from app.rag.retriever import answer_query
from app.vectorstore.qdrant_store import QdrantVectorStore


@st.cache_resource
def get_vector_store():
    return QdrantVectorStore()


def _playlist_selector() -> list[str] | None:
    try:
        playlists = get_vector_store().list_playlists()
    except Exception as e:
        st.warning(f"Qdrant is not ready yet: {e}")
        return None

    if not playlists:
        st.info("No playlists are indexed yet. Ingest a playlist first.")
        return None

    label_to_id = {
        playlist["playlist_title"]: playlist["playlist_id"]
        for playlist in playlists
    }
    selected_label = st.selectbox("Knowledge base", ["All playlists", *label_to_id.keys()])
    if selected_label == "All playlists":
        return None
    return [label_to_id[selected_label]]


def main():
    st.set_page_config(
        page_title="YouTube Playlist RAG",
        page_icon="▶️",
        layout="wide",
    )

    st.title("YouTube Playlist RAG 📚")
    st.markdown("Chat with the contents of YouTube playlists using local Whisper, embeddings, Qdrant, and Ollama.")

    tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Ingest Playlist"])

    with tab1:
        st.header("Ask a Question")
        selected_playlist_ids = _playlist_selector()

        query = st.text_input(
            "What do you want to know?",
            placeholder="e.g., What is the difference between LL and LR parsing?",
        )

        if st.button("Ask", type="primary"):
            if not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        response = answer_query(query, playlist_ids=selected_playlist_ids)

                        st.markdown("### Answer")
                        st.info(response.answer)

                        if response.sources:
                            st.markdown("### Sources")
                            for source in response.sources:
                                with st.expander(f"📍 {source.video_title} ({source.timestamp_display})"):
                                    if source.playlist_title:
                                        st.markdown(f"**Playlist:** {source.playlist_title}")
                                    st.markdown(f"**Video:** {source.video_title}")
                                    st.markdown(f"**Timestamp:** {source.timestamp_display}")
                                    st.markdown(f"[▶️ Watch from this point]({source.youtube_url})")

                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")

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
                        progress_bar.progress(min(current / total, 1.0))
                    status_text.text(f"[{current}/{total}] {msg}")

                with st.spinner("Initializing ingestion. This may take a while..."):
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
                                for failed_video in stats.failed_videos:
                                    st.write(failed_video)

                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    main()
