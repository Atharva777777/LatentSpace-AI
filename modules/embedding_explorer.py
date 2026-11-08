"""Embedding Explorer Module (U1 - Foundations).

Interactive 2D latent space visualization of sentence embeddings using
all-MiniLM-L6-v2, PCA, UMAP, and exact high-dimensional cosine similarity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from services.dimensionality_service import DimensionalityService
from services.embedding_service import EmbeddingService
from services.similarity_service import SimilarityService

# Category color map matching the dark dashboard aesthetic
CATEGORY_COLOR_MAP = {
    "Animals": "#38bdf8",  # Electric Cyan / Sky Blue
    "Technology": "#a855f7",  # Purple
    "Food": "#f59e0b",  # Amber / Orange
    "Transportation": "#10b981",  # Emerald Green
    "Sports": "#ef4444",  # Coral Red
    "Education": "#ec4899",  # Rose Pink
    "Nature": "#84cc16",  # Lime Green
    "Custom": "#f43f5e",  # Vibrant Rose for user-added items
}


@st.cache_resource(show_spinner=False)
def get_embedding_service() -> EmbeddingService:
    """Initializes and caches the singleton EmbeddingService instance."""
    return EmbeddingService()


def load_initial_dataset() -> pd.DataFrame:
    """Loads the baseline 42-sentence semantic dataset from CSV."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_sentences.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # Fallback dataset if file is absent
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "category": "Animals",
                    "sentence": "The Bengal tiger silently stalks its prey.",
                },
                {
                    "id": 2,
                    "category": "Technology",
                    "sentence": "Neural networks learn deep representations.",
                },
                {
                    "id": 3,
                    "category": "Food",
                    "sentence": "Neapolitan pizza has a blistered crust.",
                },
            ]
        )
    return df


def initialize_session_state() -> None:
    """Initializes persistent application session state variables."""
    if "corpus_df" not in st.session_state:
        st.session_state.corpus_df = load_initial_dataset()

    if "selected_sentence_idx" not in st.session_state:
        st.session_state.selected_sentence_idx = 0

    if "last_added_sentence" not in st.session_state:
        st.session_state.last_added_sentence = None


def render_embedding_explorer_page() -> None:
    """Main controller for the U1 Embedding Explorer page."""
    initialize_session_state()
    service = get_embedding_service()

    # Page Header with Title and Model Badge
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">U1: Embedding Explorer</h1>
                <p style="color: #94a3b8; margin-top: 4px; font-size: 0.95rem;">
                    High-Dimensional Semantic Projections (384-D) &bull; PCA & UMAP Dimensionality Reduction &bull; Exact Latent Cosine Search
                </p>
            </div>
            <span class="badge-pill badge-active">Active Module</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df: pd.DataFrame = st.session_state.corpus_df
    sentences = df["sentence"].tolist()
    categories = df["category"].tolist()

    # Step 1: Compute / Retrieve Embeddings
    with st.spinner("Computing high-dimensional sentence embeddings..."):
        try:
            embeddings: np.ndarray = service.encode_batch(sentences)
        except Exception as e:
            st.error(f"Failed to compute embeddings: {e}")
            return

    # Top Control Bar (Sidebar or Top Columns)
    col_config, col_stats = st.columns([2, 1])

    with col_config:
        st.markdown("##### ⚙️ Projection Algorithm")
        reduction_col, param_col = st.columns([1, 2])
        with reduction_col:
            method = st.radio(
                "Method",
                options=["PCA", "UMAP"],
                horizontal=True,
                help="PCA provides deterministic global variance axes; UMAP preserves local & global non-linear manifold topology.",
            )

        umap_params: Dict[str, Any] = {}
        with param_col:
            if method == "UMAP":
                u_col1, u_col2, u_col3 = st.columns(3)
                with u_col1:
                    umap_params["n_neighbors"] = st.slider(
                        "n_neighbors",
                        min_value=2,
                        max_value=min(30, max(2, len(sentences) - 1)),
                        value=min(15, max(2, len(sentences) - 1)),
                        help="Controls local vs global manifold balance.",
                    )
                with u_col2:
                    umap_params["min_dist"] = st.slider(
                        "min_dist",
                        min_value=0.01,
                        max_value=0.99,
                        value=0.1,
                        step=0.05,
                        help="Clustering tightness of points.",
                    )
                with u_col3:
                    umap_params["metric"] = st.selectbox(
                        "Metric",
                        options=["cosine", "euclidean", "manhattan"],
                        index=0,
                        help="Cosine metric is mathematically ideal for normalized text embeddings.",
                    )

    with col_stats:
        st.markdown("##### 📊 Corpus Statistics")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.metric("Sentences", len(df))
        with s_col2:
            st.metric("Dimensions", service.get_embedding_dimension())
        with s_col3:
            st.metric("Categories", df["category"].nunique())

    # Step 2: Dimensionality Reduction
    try:
        coords, meta = DimensionalityService.reduce(
            embeddings,
            method=method,
            n_components=2,
            **umap_params,
        )
    except Exception as e:
        st.error(f"Dimensionality reduction error: {e}")
        return

    # Merge coordinates into DataFrame for visualization
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["hover_text"] = [
        f"<b>[{cat}]</b><br>{text}" for cat, text in zip(plot_df["category"], plot_df["sentence"])
    ]

    # Display Variance Info for PCA
    if method == "PCA":
        pc1_pct = meta["pc1_variance"] * 100
        pc2_pct = meta["pc2_variance"] * 100
        tot_pct = meta["total_variance_explained"] * 100
        st.caption(
            f"📈 **PCA Variance Explained**: PC1 = `{pc1_pct:.2f}%` | PC2 = `{pc2_pct:.2f}%` | **Cumulative Total** = `{tot_pct:.2f}%`"
        )
    elif method == "UMAP":
        st.caption(
            f"🌐 **UMAP Manifold Config**: Neighbors = `{meta['n_neighbors']}` | Min Dist = `{meta['min_dist']}` | Metric = `{meta['metric']}`"
        )

    # Step 3: Interactive Visualisation and Selection
    viz_col, inspect_col = st.columns([1.5, 1])

    with viz_col:
        st.markdown("##### 🌌 Latent Space Scatter Projection")

        # Dropdown sentence selector to synchronize selection
        current_idx = st.session_state.selected_sentence_idx
        if current_idx >= len(plot_df):
            current_idx = 0
            st.session_state.selected_sentence_idx = 0

        # Build Plotly Figure
        fig = go.Figure()

        # Add traces for each category
        for cat in sorted(plot_df["category"].unique()):
            cat_df = plot_df[plot_df["category"] == cat]
            color = CATEGORY_COLOR_MAP.get(cat, "#38bdf8")

            fig.add_trace(
                go.Scatter(
                    x=cat_df["x"],
                    y=cat_df["y"],
                    mode="markers",
                    name=cat,
                    text=cat_df["hover_text"],
                    hoverinfo="text",
                    customdata=cat_df.index,
                    marker=dict(
                        size=11,
                        color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.4)"),
                        opacity=0.85,
                    ),
                )
            )

        # Highlight Selected Point
        selected_row = plot_df.iloc[current_idx]
        fig.add_trace(
            go.Scatter(
                x=[selected_row["x"]],
                y=[selected_row["y"]],
                mode="markers",
                name="Selected Sentence",
                hoverinfo="text",
                text=[f"⭐ <b>SELECTED</b><br>[{selected_row['category']}] {selected_row['sentence']}"],
                marker=dict(
                    size=22,
                    color="#facc15",  # Bright Gold
                    symbol="star-diamond",
                    line=dict(width=2.5, color="#ffffff"),
                ),
                showlegend=True,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(18, 24, 36, 0.7)",
            plot_bgcolor="rgba(14, 18, 28, 0.9)",
            height=540,
            margin=dict(l=20, r=20, t=30, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(18, 24, 36, 0.6)",
                bordercolor="#243247",
                borderwidth=1,
            ),
            xaxis=dict(
                title=f"{method} Dimension 1",
                gridcolor="rgba(36, 50, 71, 0.4)",
                zerolinecolor="rgba(36, 50, 71, 0.8)",
            ),
            yaxis=dict(
                title=f"{method} Dimension 2",
                gridcolor="rgba(36, 50, 71, 0.4)",
                zerolinecolor="rgba(36, 50, 71, 0.8)",
            ),
            hoverlabel=dict(
                bgcolor="#1e293b",
                font_size=12,
                font_family="Inter",
            ),
        )

        # Render with on_select callback if available in Streamlit
        try:
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="latent_scatter_plot",
            )
            # Detect click on chart
            if event and "selection" in event and event["selection"].get("points"):
                clicked_points = event["selection"]["points"]
                if clicked_points:
                    point = clicked_points[0]
                    # Retrieve point index from customdata or point_index
                    if "customdata" in point:
                        st.session_state.selected_sentence_idx = int(point["customdata"])
                    elif "point_index" in point:
                        # Fallback
                        st.session_state.selected_sentence_idx = int(point["point_index"])
        except TypeError:
            # Fallback for Streamlit versions without on_select parameter
            st.plotly_chart(fig, use_container_width=True)

    with inspect_col:
        st.markdown("##### 🎯 Selected Point & Nearest Neighbours")

        # Sentence Selector Sync Dropdown
        options = [f"#{i+1} [{row['category']}] {row['sentence'][:45]}..." for i, row in plot_df.iterrows()]
        chosen_opt = st.selectbox(
            "Select sentence to inspect:",
            options=options,
            index=current_idx,
            help="Select any sentence in the corpus to retrieve its original latent vector and compute semantic nearest neighbours.",
        )
        selected_from_dropdown = options.index(chosen_opt)
        if selected_from_dropdown != current_idx:
            st.session_state.selected_sentence_idx = selected_from_dropdown
            st.rerun()

        # Display Selected Sentence Card
        target_sentence = sentences[current_idx]
        target_category = categories[current_idx]
        target_coords = (float(plot_df.iloc[current_idx]["x"]), float(plot_df.iloc[current_idx]["y"]))

        st.markdown(
            f"""
            <div class="research-card" style="border-left: 4px solid #facc15; padding: 14px 18px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #facc15; font-weight: 700; letter-spacing: 0.05em;">
                    Active Reference Query &bull; #{current_idx + 1}
                </div>
                <div style="font-size: 1.0rem; font-weight: 600; color: #f8fafc; margin-top: 6px;">
                    "{target_sentence}"
                </div>
                <div style="display: flex; gap: 10px; margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                    <span><b>Category:</b> {target_category}</span>
                    <span>&bull;</span>
                    <span><b>2D Coords:</b> ({target_coords[0]:.3f}, {target_coords[1]:.3f})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 4: Nearest Neighbours Calculation in 384-D Latent Space
        st.markdown(
            "###### 🔗 Top 5 Semantic Nearest Neighbours "
            "<span style='font-size: 0.75rem; color: #38bdf8;'>(Evaluated in 384-D Latent Space)</span>",
            unsafe_allow_html=True,
        )

        neighbors = SimilarityService.get_top_k_neighbors(
            query_idx=current_idx,
            embeddings=embeddings,
            sentences=sentences,
            categories=categories,
            top_k=5,
            exclude_self=True,
        )

        for n in neighbors:
            sim_val = n["similarity"]
            pct = n["percentage"]
            cat_name = n["category"]
            cat_color = CATEGORY_COLOR_MAP.get(cat_name, "#38bdf8")

            st.markdown(
                f"""
                <div class="neighbor-card">
                    <div style="flex: 1; padding-right: 12px;">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                            <span style="font-weight: 700; color: #94a3b8; font-size: 0.75rem;">#{n['rank']}</span>
                            <span style="background: {cat_color}22; color: {cat_color}; border: 1px solid {cat_color}44; 
                                         padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">
                                {cat_name}
                            </span>
                        </div>
                        <div class="neighbor-text">{n['sentence']}</div>
                    </div>
                    <div class="neighbor-score">
                        <div>{sim_val:.4f}</div>
                        <div style="font-size: 0.7rem; color: #94a3b8;">{pct:.1f}% match</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Step 5: Custom Sentence Addition & Latent Probe
    st.markdown("### ✍️ Feature: Add Custom Sentences & Latent Probing")

    tab_add, tab_probe, tab_inspect = st.tabs(
        ["➕ Add to Corpus", "🔬 Instant Vector Probe", "📋 Data Inspector & Coordinates"]
    )

    with tab_add:
        st.markdown(
            "Add a custom sentence to the dataset. The application will compute its real 384-dimensional "
            "Sentence-BERT embedding, re-project the entire latent space, and locate its position among other concepts."
        )

        with st.form("add_sentence_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                new_sentence = st.text_input(
                    "Sentence",
                    placeholder="e.g., Deep convolutional networks extract hierarchical visual features.",
                )
            with f_col2:
                existing_cats = sorted(list(set(categories))) + ["Custom"]
                new_category = st.selectbox("Category", options=existing_cats)

            submit_btn = st.form_submit_button("Add Sentence & Re-Project")

            if submit_btn:
                clean_text = new_sentence.strip()
                if not clean_text:
                    st.warning("⚠️ Input sentence cannot be empty or only whitespace.")
                elif clean_text in sentences:
                    st.warning(f"⚠️ This sentence is already in the dataset at index #{sentences.index(clean_text) + 1}.")
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "id": len(df) + 1,
                                "category": new_category,
                                "sentence": clean_text,
                            }
                        ]
                    )
                    st.session_state.corpus_df = pd.concat([df, new_row], ignore_index=True)
                    st.session_state.selected_sentence_idx = len(df)  # Select the newly added item
                    st.session_state.last_added_sentence = clean_text
                    st.success(f"✅ Added sentence #{len(df) + 1}! Re-projecting latent space...")
                    st.rerun()

    with tab_probe:
        st.markdown(
            "Probe the 384-dimensional latent space with an arbitrary query text **without modifying the corpus**. "
            "Calculates exact cosine similarity against all current corpus sentences."
        )
        probe_query = st.text_input(
            "Query Probe Text",
            placeholder="e.g., Italian culinary pizza with melted mozzarella cheese.",
            key="probe_input",
        )
        if probe_query and probe_query.strip():
            try:
                probe_vec = service.encode_sentence(probe_query.strip())
                probe_neighbors = SimilarityService.get_top_k_neighbors_for_vector(
                    query_vector=probe_vec,
                    embeddings=embeddings,
                    sentences=sentences,
                    categories=categories,
                    top_k=5,
                )

                st.markdown(f"**Results for:** *\"{probe_query.strip()}\"*")
                p_cols = st.columns(5)
                for i, n in enumerate(probe_neighbors):
                    with p_cols[i]:
                        st.markdown(
                            f"""
                            <div class="metric-box" style="height: 100%;">
                                <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">#{n['rank']} ({n['similarity']:.4f})</div>
                                <div style="font-size: 0.7rem; color: #94a3b8; margin: 4px 0;">{n['category']}</div>
                                <div style="font-size: 0.8rem; color: #f1f5f9; text-align: left;">{n['sentence']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Probe calculation error: {e}")

    with tab_inspect:
        st.markdown("##### 📊 Full Corpus Coordinates & Embeddings Inspector")
        inspect_df = plot_df[["id", "category", "sentence", "x", "y"]].copy()
        inspect_df.columns = ["ID", "Category", "Sentence", f"{method}_X", f"{method}_Y"]
        st.dataframe(
            inspect_df.style.format({f"{method}_X": "{:.4f}", f"{method}_Y": "{:.4f}"}),
            use_container_width=True,
            height=300,
        )

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🔄 Reset to Default Dataset", help="Reverts custom sentences back to initial 42 sentences"):
                st.session_state.corpus_df = load_initial_dataset()
                st.session_state.selected_sentence_idx = 0
                st.success("Corpus reset to default 42 sentences.")
                st.rerun()

    st.divider()

    # Step 6: Educational Section (Feature 7)
    render_educational_section()


def render_educational_section() -> None:
    """Renders the comprehensive viva-ready educational breakdown."""
    st.markdown("### 🎓 Educational Insights & Theory (Viva Preparation)")

    with st.expander("📖 1. What are Embeddings & Why are They Necessary?", expanded=False):
        st.markdown(
            """
            - **Definition**: Embeddings are dense, continuous, low-dimensional vector representations 
              $\\mathbf{e} \\in \\mathbb{R}^D$ (where $D=384$ for `all-MiniLM-L6-v2`) that capture 
              latent semantic and syntactic relationships of natural language.
            - **Semantic Geometry**: Sentences conveying similar meanings are mapped to vectors with small angles 
              (high cosine similarity) in this continuous space, solving the lexical mismatch problem 
              of traditional sparse bag-of-words or TF-IDF models.
            - **Curse of Dimensionality**: High-dimensional spaces ($D > 300$) suffer from extreme volume growth 
              and geometric sparsity. Humans cannot visually comprehend dimensions beyond 3D, making dimensionality 
              reduction essential for exploratory data analysis, interpretability, and sanity checking.
            """
        )

    with st.expander("🔬 2. PCA vs. UMAP: Understanding the Mathematical Trade-Offs", expanded=False):
        st.markdown(
            """
            | Property | Principal Component Analysis (PCA) | Uniform Manifold Approximation & Projection (UMAP) |
            | :--- | :--- | :--- |
            | **Nature** | Linear, deterministic orthogonal transformation | Non-linear, manifold learning technique |
            | **Objective** | Maximizes global variance $\\max_{\\mathbf{w}} \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}$ | Minimizes fuzzy topological cross-entropy |
            | **Preservation** | Preserves global Euclidean pairwise distances | Preserves local neighborhood graph structure |
            | **Hyperparameters** | None (purely algebraic eigen-decomposition) | `n_neighbors`, `min_dist`, distance metric |
            | **Speed** | Extremely fast via SVD: $\\mathcal{O}(N D^2)$ | Fast approximate nearest neighbor search via PyNNDescent |
            | **Ideal For** | Baseline variance audit, quick global sanity checks | Clustering visualization, complex semantic topologies |
            """
        )

    with st.expander("📐 3. How Cosine Similarity Works (Why High-D Matters)", expanded=False):
        st.markdown(
            """
            The cosine similarity between two embedding vectors $\\mathbf{u}$ and $\\mathbf{v}$ is defined as:

            $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \\frac{\\sum_{i=1}^D u_i v_i}{\\sqrt{\\sum_{i=1}^D u_i^2} \\sqrt{\\sum_{i=1}^D v_i^2}} = \\cos(\\theta)$$

            - **Range**: $[-1.0, 1.0]$, where $1.0$ indicates identical orientation, $0.0$ orthogonal, and $-1.0$ diametrically opposite.
            - **Scale Invariance**: Length (magnitude) of the vector is normalized out; only the angle $\\theta$ matters.
            - **CRITICAL ARCHITECTURAL DISTINCTION**: In **LATENTSPACE AI**, nearest-neighbour retrieval is computed on the 
              **original 384-dimensional embeddings**. Dimensionality reduction to 2D is strictly an illustrative visualization 
              tool; projecting 384 dimensions to 2 dimensions causes unavoidable geometric distortion (Johnson-Lindenstrauss lemma). 
              Computing similarity on 2D coordinates would introduce significant false neighbors.
            """
        )

    with st.expander("💡 4. University Viva Q&A Quick Reference", expanded=False):
        st.markdown(
            r"""
            1. **Q: Why use `all-MiniLM-L6-v2` instead of full BERT base?**  
               *A:* `all-MiniLM-L6-v2` was distilled using MiniLM knowledge distillation into a 6-layer transformer with 384 hidden dimensions. It retains 99% of BERT's semantic performance while being 5x faster and consuming under 120MB of memory, making it ideal for low-latency interactive inference on CPU.

            2. **Q: What is the Johnson-Lindenstrauss lemma and why does it matter here?**  
               *A:* The Johnson-Lindenstrauss lemma states that a collection of points in high-dimensional space can be embedded into a space of much lower dimension while approximately preserving pairwise distances, but the minimum dimension required scales logarithmically with the number of points ($\mathcal{O}(\log N / \epsilon^2)$). Reducing 384 dimensions to just 2 causes significant metric distortion, which is why nearest-neighbour retrieval MUST use the unreduced 384-D vectors.

            3. **Q: Why is UMAP with cosine metric preferred over Euclidean for sentence embeddings?**  
               *A:* Sentence transformer embeddings reside on the surface of a high-dimensional unit hypersphere. Euclidean distance in high dimensions suffers from distance concentration (all points tend to be equidistant from one another). Cosine distance directly measures angular separation on the hypersphere manifold.
            """
        )

