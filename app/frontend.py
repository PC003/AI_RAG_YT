"""Streamlit frontend for YouTube Playlist RAG."""

import streamlit as st
from app.main import YouTubeRAGOrchestrator
from app.config import QDRANT_COLLECTION
from app.utils.timestamps import seconds_to_hms

# Page config
st.set_page_config(
    page_title="YouTube Playlist RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .timestamp-link {
        color: #1a73e8;
        text-decoration: none;
        font-weight: 500;
    }
    .timestamp-link:hover {
        text-decoration: underline;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #d0e3f0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_orchestrator():
    """Get or create the orchestrator instance."""
    if "orchestrator" not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.orchestrator = YouTubeRAGOrchestrator()
    return st.session_state.orchestrator


def main():
    st.title("🎬 YouTube Playlist RAG")
    st.markdown("**Chat with your YouTube playlists** - Ask questions and get answers with timestamped sources.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        try:
            orch = get_orchestrator()
            stats = orch.get_stats()
            st.metric("Collection", stats.get("collection_name", QDRANT_COLLECTION))
            st.metric("Chunks Stored", stats.get("vector_count", 0))
            st.success("System Ready")
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")
            return

        st.divider()
        st.markdown("""
        ### 📋 How to Use
        1. Enter a YouTube playlist URL
        2. Click 'Process Playlist'
        3. Wait for processing to complete
        4. Ask questions about the videos
        5. Click source links to jump to timestamps
        """)

    # Main content area
    tab1, tab2 = st.tabs(["📚 Process Playlist", "💬 Chat"])

    # Tab 1: Playlist Processing
    with tab1:
        st.header("Process a YouTube Playlist")

        playlist_url = st.text_input(
            "YouTube Playlist URL",
            placeholder="https://www.youtube.com/playlist?list=...",
            help="Enter the full URL of a public YouTube playlist"
        )

        if st.button("🚀 Process Playlist", type="primary", disabled=not playlist_url):
            if not playlist_url.strip():
                st.error("Please enter a playlist URL")
            elif "playlist" not in playlist_url and "list=" not in playlist_url:
                st.warning("This doesn't look like a playlist URL. It should contain 'playlist' or 'list=' parameter.")
            else:
                with st.spinner("Processing playlist... This may take a while for large playlists."):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    try:
                        orch = get_orchestrator()
                        result = orch.process_playlist(playlist_url)

                        progress_bar.progress(100)

                        # Display results
                        st.success("✅ Playlist processing complete!")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Videos Found", result.videos_found)
                        col2.metric("Processed", result.videos_processed)
                        col3.metric("Skipped", result.videos_skipped)
                        col4.metric("Chunks Created", result.chunks_created)

                        if result.videos_failed > 0:
                            st.warning(f"⚠️ {result.videos_failed} videos failed to process")
                            with st.expander("View failed videos"):
                                for vid in result.failed_videos:
                                    st.text(vid)

                    except Exception as e:
                        st.error(f"Error processing playlist: {str(e)}")

    # Tab 2: Chat
    with tab2:
        st.header("Ask a Question")

        # Check if any data is available
        orch = get_orchestrator()
        stats = orch.get_stats()

        if stats.get("vector_count", 0) == 0:
            st.info("📌 No data available yet. Process a playlist first to enable chat.")
            return

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # Display answer
                    st.markdown(f'<div class="answer-box">{message["content"]["answer"]}</div>', unsafe_allow_html=True)

                    # Display sources
                    if message["content"].get("sources"):
                        st.markdown("#### 📺 Sources")
                        for source in message["content"]["sources"]:
                            with st.container():
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>{source['video_title']}</strong><br>
                                    ⏱️ <a href="{source['youtube_url']}" target="_blank" class="timestamp-link">{source['timestamp_display']}</a>
                                </div>
                                """, unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask a question about the videos..."):
            # Display user message
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = orch.answer_query(prompt)

                        # Display answer
                        st.markdown(f'<div class="answer-box">{response.answer}</div>', unsafe_allow_html=True)

                        # Display sources
                        if response.sources:
                            st.markdown("#### 📺 Sources")
                            for source in response.sources:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>{source.video_title}</strong><br>
                                        ⏱️ <a href="{source.youtube_url}" target="_blank" class="timestamp-link">{source.timestamp_display}</a>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": {
                                "answer": response.answer,
                                "sources": [s.model_dump() for s in response.sources]
                            }
                        })

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()
