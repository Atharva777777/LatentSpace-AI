"""LATENTSPACE AI: Multimodal Intelligence Explorer.

Generative AI CA Mini Project - Problem Statement P6.
Entry point application for Streamlit dashboard.
"""

from pathlib import Path
import streamlit as st
from modules.bias_analytics import render_bias_analytics_page
from modules.cluster_intelligence import render_cluster_intelligence_page
from modules.embedding_explorer import render_embedding_explorer_page
from modules.generative_studio import render_generative_studio_page
from modules.multimodal_explorer import render_multimodal_explorer_page

# Configure Streamlit page settings
st.set_page_config(
    page_title="LATENTSPACE AI | Multimodal Intelligence Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Loads and injects custom CSS for modern dark research theme."""
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """Renders the persistent sidebar with app metadata and unit navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 12px; border-bottom: 1px solid #243247; margin-bottom: 16px;">
                <div style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    LATENTSPACE AI
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                    Multimodal Intelligence Explorer
                </div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 4px;">
                    Generative AI Mini Project &bull; Problem P6
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Modules Navigation")

        nav_options = [
            "U1: Embedding Explorer",
            "U2: Cluster Intelligence",
            "U3: Multimodal Explorer",
            "U4: Generative Studio",
            "U5: Bias Analytics",
        ]

        selection = st.radio(
            "Select Module",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Engine Architecture Diagnostics
        st.markdown("### 🖥️ Engine Status")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
                <div><b>Framework:</b> Streamlit 1.30+</div>
                <div><b>Active Model:</b> all-MiniLM-L6-v2</div>
                <div><b>Latent Dim:</b> 384 Dimensions</div>
                <div><b>Execution Mode:</b> Local CPU</div>
                <div><b>Active Module:</b> U1 (Foundations)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 24px; padding: 12px; background: rgba(36, 50, 71, 0.3); border-radius: 8px; border: 1px solid #243247; font-size: 0.72rem; color: #64748b;">
                Developed as an autonomous GenAI research framework for Latent-Space Exploration.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selection


def main() -> None:
    """Main application lifecycle router."""
    inject_custom_styles()
    module = render_sidebar()

    if module == "U1: Embedding Explorer":
        render_embedding_explorer_page()
    elif module == "U2: Cluster Intelligence":
        render_cluster_intelligence_page()
    elif module == "U3: Multimodal Explorer":
        render_multimodal_explorer_page()
    elif module == "U4: Generative Studio":
        render_generative_studio_page()
    elif module == "U5: Bias Analytics":
        render_bias_analytics_page()


if __name__ == "__main__":
    main()

