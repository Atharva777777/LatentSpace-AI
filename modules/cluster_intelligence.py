"""Cluster Intelligence Module (U2 - GANs & LLMs).

Performs K-Means semantic clustering in 384-D embedding space,
uses an LLM (Flan-T5) to auto-synthesize descriptive cluster names,
and trains a PyTorch Variational Autoencoder (VAE) on Fashion-MNIST.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from services.clustering_service import ClusteringService
from services.dimensionality_service import DimensionalityService
from services.embedding_service import EmbeddingService
from services.llm_naming_service import LLMNamingService
from services.vae_service import VAEService

# High-contrast color palette for cluster visual separation
CLUSTER_COLORS = [
    "#38bdf8",  # Electric Cyan
    "#a855f7",  # Purple
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#ef4444",  # Coral
    "#ec4899",  # Pink
    "#84cc16",  # Lime
    "#06b6d4",  # Teal
    "#f43f5e",  # Rose
    "#6366f1",  # Indigo
]


@st.cache_resource(show_spinner=False)
def get_services() -> Dict[str, Any]:
    """Caches service singletons."""
    return {
        "embedding": EmbeddingService(),
        "naming": LLMNamingService(),
        "vae": VAEService(),
    }


def render_cluster_intelligence_page() -> None:
    """Main controller for Unit 2: Cluster Intelligence & VAE Training."""
    svcs = get_services()
    embed_svc: EmbeddingService = svcs["embedding"]
    naming_svc: LLMNamingService = svcs["naming"]
    vae_svc: VAEService = svcs["vae"]

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">U2: Cluster Intelligence & VAE Training</h1>
                <p style="color: #94a3b8; margin-top: 4px; font-size: 0.95rem;">
                    LLM-Driven Semantic Cluster Auto-Naming &bull; PyTorch Fashion-MNIST Variational Autoencoder
                </p>
            </div>
            <span class="badge-pill badge-active">Active Module</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_clustering, tab_vae = st.tabs(["🏷️ LLM Cluster Naming", "🎨 Fashion-MNIST VAE Training"])

    # =========================================================================
    # TAB 1: LLM CLUSTER NAMING
    # =========================================================================
    with tab_clustering:
        st.markdown(
            "Partition the 384-dimensional sentence space into clusters and use an instruction-tuned "
            "LLM (`google/flan-t5-small`) to automatically synthesize descriptive topic titles from centroid exemplars."
        )

        # Retrieve corpus from session_state or default
        if "corpus_df" not in st.session_state:
            from modules.embedding_explorer import load_initial_dataset
            st.session_state.corpus_df = load_initial_dataset()

        df: pd.DataFrame = st.session_state.corpus_df
        sentences = df["sentence"].tolist()
        categories = df["category"].tolist()

        c_col1, c_col2, c_col3 = st.columns([1, 1, 2])
        with c_col1:
            n_clusters = st.slider("Number of Clusters (k)", min_value=3, max_value=10, value=7)
        with c_col2:
            use_llm = st.checkbox("Enable Local LLM (Flan-T5)", value=True, help="Synthesizes names via Seq2Seq prompt")
        with c_col3:
            run_btn = st.button("🚀 Run Clustering & Auto-Naming", type="primary")

        # Run or retrieve clustering
        if run_btn or "cluster_results" not in st.session_state:
            with st.spinner("Computing 384-D sentence embeddings..."):
                embeddings = embed_svc.encode_batch(sentences)

            with st.spinner(f"Clustering into {n_clusters} semantic partitions..."):
                labels, centroids, sil_score = ClusteringService.cluster_embeddings(
                    embeddings, n_clusters=n_clusters
                )
                summaries = ClusteringService.extract_cluster_summaries(
                    embeddings, labels, centroids, sentences, categories
                )

            with st.spinner("Prompting LLM to synthesize cluster names..."):
                for summary in summaries:
                    summary["llm_name"] = naming_svc.name_cluster(
                        exemplars=summary["exemplars"],
                        keywords=summary["keywords"],
                        use_llm=use_llm,
                    )

            # Store in session state
            st.session_state.cluster_results = {
                "labels": labels,
                "centroids": centroids,
                "sil_score": sil_score,
                "summaries": summaries,
                "embeddings": embeddings,
            }

        res = st.session_state.cluster_results
        labels = res["labels"]
        summaries = res["summaries"]
        sil_score = res["sil_score"]
        embeddings = res["embeddings"]

        # Reduce to 2D for visualization
        coords, _ = DimensionalityService.reduce(embeddings, method="PCA", n_components=2)
        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["cluster_id"] = labels

        # Map cluster name
        cluster_name_map = {s["cluster_id"]: f"C{s['cluster_id']}: {s['llm_name']}" for s in summaries}
        plot_df["cluster_label"] = plot_df["cluster_id"].map(cluster_name_map)

        # Plotly Cluster Visualization
        col_plot, col_cards = st.columns([1.5, 1])

        with col_plot:
            st.markdown(f"##### 🌌 2D Projection Colored by LLM Cluster Topic (Silhouette: `{sil_score:.3f}`)")
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                color="cluster_label",
                hover_data=["sentence", "category"],
                color_discrete_sequence=CLUSTER_COLORS,
                template="plotly_dark",
                labels={"x": "PCA Dim 1", "y": "PCA Dim 2", "cluster_label": "LLM Cluster"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(18, 24, 36, 0.7)",
                plot_bgcolor="rgba(14, 18, 28, 0.9)",
                height=520,
                margin=dict(l=20, r=20, t=30, b=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_cards:
            st.markdown("##### 🏷️ LLM Synthesized Topics & Exemplars")
            for s in summaries:
                c_id = s["cluster_id"]
                c_color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                st.markdown(
                    f"""
                    <div class="research-card" style="border-left: 4px solid {c_color}; padding: 12px 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: {c_color}; font-size: 0.95rem;">
                                Cluster {c_id}: {s['llm_name']}
                            </span>
                            <span style="font-size: 0.75rem; color: #94a3b8;">{s['count']} items</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">
                            <b>Keywords:</b> {', '.join(s['keywords']) if s['keywords'] else 'N/A'}
                        </div>
                        <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;">
                            <b>Top Exemplar:</b> "{s['exemplars'][0] if s['exemplars'] else ''}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 2: FASHION-MNIST VAE TRAINING
    # =========================================================================
    with tab_vae:
        st.markdown(
            "Train a continuous 2D Variational Autoencoder on the **Fashion-MNIST** dataset. "
            "The model minimizes the Evidence Lower Bound (ELBO): Reconstruction loss + $\\beta \\cdot D_{KL}$. "
            "Once trained, the checkpoint is automatically preserved for interactive point decoding in **U4: Generative Studio**."
        )

        v_col1, v_col2, v_col3 = st.columns([1, 1, 1])
        with v_col1:
            epochs = st.slider("Training Epochs", min_value=2, max_value=12, value=6)
        with v_col2:
            subset_size = st.select_slider(
                "Training Subset (Images)",
                options=[1000, 2000, 3000, 5000],
                value=3000,
            )
        with v_col3:
            beta = st.slider("KL Weight (β)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

        train_btn = st.button("⚡ Train Fashion-MNIST VAE", type="primary")

        # Check if pre-trained checkpoint exists
        if vae_svc.is_trained() and "vae_history" not in st.session_state:
            vae_svc.load_checkpoint()
            st.session_state.vae_trained = True

        if train_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def on_epoch(epoch: int, total_epochs: int, metrics: Dict[str, float]):
                progress_bar.progress(epoch / total_epochs)
                status_text.text(
                    f"Epoch {epoch}/{total_epochs} &bull; Total Loss: {metrics['loss']:.2f} "
                    f"&bull; BCE: {metrics['bce']:.2f} &bull; KL: {metrics['kl']:.2f}"
                )

            with st.spinner("Training Fashion-MNIST VAE on CPU..."):
                history = vae_svc.train_vae(
                    epochs=epochs,
                    subset_size=subset_size,
                    beta=beta,
                    progress_callback=on_epoch,
                )
                st.session_state.vae_history = history
                st.session_state.vae_trained = True
                status_text.success("✅ VAE Training complete! Checkpoint saved to `models/fashion_mnist_vae.pt`")

        # Display Training History & Reconstructions if trained
        if getattr(st.session_state, "vae_trained", False) or vae_svc.is_trained():
            vae_svc.load_checkpoint()

            disp_col1, disp_col2 = st.columns([1, 1])

            with disp_col1:
                st.markdown("##### 📉 ELBO Loss Convergence")
                if "vae_history" in st.session_state:
                    hist_df = pd.DataFrame(st.session_state.vae_history)
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["loss"], name="Total ELBO", line=dict(color="#38bdf8", width=2)))
                    fig_loss.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["bce"], name="BCE Recon", line=dict(color="#10b981", width=1.5)))
                    fig_loss.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["kl"], name="KL Divergence", line=dict(color="#f59e0b", width=1.5)))
                    fig_loss.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(18, 24, 36, 0.7)",
                        plot_bgcolor="rgba(14, 18, 28, 0.9)",
                        height=320,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Epoch",
                        yaxis_title="Loss per Sample",
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.info("Loaded pre-trained VAE checkpoint from disk (`models/fashion_mnist_vae.pt`).")

            with disp_col2:
                st.markdown("##### 🖼️ VAE Reconstruction Quality (Test Images)")
                try:
                    origs, recons, class_labels = vae_svc.get_sample_reconstructions(num_samples=4)
                    for i in range(4):
                        r_c1, r_c2, r_c3 = st.columns([1, 1, 2])
                        with r_c1:
                            st.image(origs[i], caption=f"Original: {class_labels[i]}", width=70, clamp=True)
                        with r_c2:
                            st.image(recons[i], caption=f"Reconstructed", width=70, clamp=True)
                        with r_c3:
                            st.markdown(f"<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 15px;'>Class: <b>{class_labels[i]}</b><br>Latent: Continuous $\\mathbf{{z}} \\in \\mathbb{{R}}^2$</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not load sample reconstructions: {e}")
