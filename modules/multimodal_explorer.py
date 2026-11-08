"""Multimodal Explorer Module (U3 - GPT, BERT & Multimodal).

Embeds text with Sentence-BERT/CLIP and images with CLIP into a shared
multimodal latent space visualized in an interactive 2D joint projection.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from services.multimodal_service import MultimodalService

# Standard apparel classes and curated textual descriptions
APPAREL_ITEMS = [
    {"class": "T-shirt/top", "prompt": "A casual short-sleeved cotton t-shirt", "color": "#38bdf8"},
    {"class": "Trouser", "prompt": "Pair of formal denim trousers or jeans", "color": "#818cf8"},
    {"class": "Pullover", "prompt": "A warm knitted wool pullover sweater", "color": "#c084fc"},
    {"class": "Dress", "prompt": "An elegant evening dress for women", "color": "#f43f5e"},
    {"class": "Coat", "prompt": "A heavy winter overcoat outerwear jacket", "color": "#f59e0b"},
    {"class": "Sandal", "prompt": "Open-toe summer beach sandals footwear", "color": "#10b981"},
    {"class": "Shirt", "prompt": "A collared button-down dress shirt", "color": "#06b6d4"},
    {"class": "Sneaker", "prompt": "Sport athletic running sneaker shoes", "color": "#e11d48"},
    {"class": "Bag", "prompt": "A leather handbag tote shoulder bag", "color": "#84cc16"},
    {"class": "Ankle boot", "prompt": "High ankle leather boots with sturdy soles", "color": "#d946ef"},
]


@st.cache_resource(show_spinner=False)
def get_multimodal_service() -> MultimodalService:
    """Caches MultimodalService singleton."""
    return MultimodalService()


@st.cache_data(show_spinner=False)
def load_sample_images_and_texts() -> Tuple[List[Image.Image], List[str], List[str], List[str]]:
    """Loads representative test images from Fashion-MNIST and matched text prompts."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"
    transform = transforms.ToTensor()
    test_dataset = datasets.FashionMNIST(root=str(data_dir), train=False, download=False, transform=transform)

    # Find one representative image per class (0-9)
    found_images: Dict[int, Image.Image] = {}
    for img_tensor, label in test_dataset:
        if label not in found_images:
            # Convert 1x28x28 tensor to 28x28 grayscale PIL Image, then upscale to 56x56 RGB for CLIP
            np_img = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img, mode="L").convert("RGB").resize((56, 56))
            found_images[label] = pil_img
        if len(found_images) == 10:
            break

    images: List[Image.Image] = []
    image_classes: List[str] = []
    text_prompts: List[str] = []
    text_classes: List[str] = []

    for item in APPAREL_ITEMS:
        cls_name = item["class"]
        # Find index of class in VAEService.FASHION_CLASSES
        class_idx = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ].index(cls_name)

        if class_idx in found_images:
            images.append(found_images[class_idx])
            image_classes.append(cls_name)
            text_prompts.append(item["prompt"])
            text_classes.append(cls_name)

    return images, image_classes, text_prompts, text_classes


def render_multimodal_explorer_page() -> None:
    """Main controller for Unit 3: Multimodal Latent Space Explorer."""
    mm_svc = get_multimodal_service()

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">U3: Multimodal Latent Space Explorer</h1>
                <p style="color: #94a3b8; margin-top: 4px; font-size: 0.95rem;">
                    Joint Text & Vision Representations &bull; OpenAI CLIP (ViT-B/32) 512-D Shared Manifold
                </p>
            </div>
            <span class="badge-pill badge-active">Active Module</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading multimodal test assets and computing CLIP embeddings..."):
        try:
            images, img_classes, texts, text_classes = load_sample_images_and_texts()
            text_embs = mm_svc.encode_text(texts)
            img_embs = mm_svc.encode_images(images)
        except Exception as e:
            st.error(f"Error initializing multimodal models: {e}")
            return

    # Projection Method selector
    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        proj_method = st.radio("Joint Projection", options=["PCA", "UMAP"], horizontal=True)

    text_2d, img_2d, _ = mm_svc.project_shared_space(
        text_embs, img_embs, method=proj_method
    )

    # Build joint Plotly scatter figure
    fig = go.Figure()

    # Add Text traces (Circles)
    fig.add_trace(
        go.Scatter(
            x=text_2d[:, 0],
            y=text_2d[:, 1],
            mode="markers+text",
            name="📝 Text Descriptions",
            text=[f"\"{t[:24]}...\"" for t in texts],
            textposition="top center",
            hovertext=[f"<b>[TEXT PROMPT]</b><br>Class: {c}<br>\"{t}\"" for c, t in zip(text_classes, texts)],
            hoverinfo="text",
            marker=dict(
                size=14,
                symbol="circle",
                color="#38bdf8",
                line=dict(width=2, color="#ffffff"),
            ),
        )
    )

    # Add Image traces (Diamonds)
    fig.add_trace(
        go.Scatter(
            x=img_2d[:, 0],
            y=img_2d[:, 1],
            mode="markers+text",
            name="🖼️ Apparel Images",
            text=[f"Img: {c}" for c in img_classes],
            textposition="bottom center",
            hovertext=[f"<b>[IMAGE SAMPLE]</b><br>Class: {c}" for c in img_classes],
            hoverinfo="text",
            marker=dict(
                size=18,
                symbol="diamond",
                color="#f59e0b",
                line=dict(width=2, color="#ffffff"),
            ),
        )
    )

    # Draw dashed connecting lines between matching text-image pairs to visualize semantic grounding!
    for i in range(len(texts)):
        fig.add_trace(
            go.Scatter(
                x=[text_2d[i, 0], img_2d[i, 0]],
                y=[text_2d[i, 1], img_2d[i, 1]],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(148, 163, 184, 0.35)", width=1.5, dash="dot"),
                hoverinfo="none",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(18, 24, 36, 0.7)",
        plot_bgcolor="rgba(14, 18, 28, 0.9)",
        height=540,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title=f"{proj_method} Dimension 1",
        yaxis_title=f"{proj_method} Dimension 2",
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
    )

    v_col1, v_col2 = st.columns([1.6, 1])

    with v_col1:
        st.markdown(f"##### 🌐 Shared Latent Space: Text & Images Co-Located ({proj_method})")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dotted lines connect semantically paired text descriptions and their corresponding apparel images.")

    with v_col2:
        st.markdown("##### 🔬 Cross-Modal Cosine Alignment")
        sim_matrix = mm_svc.compute_cross_modal_similarity(text_embs, img_embs)
        sim_df = pd.DataFrame(sim_matrix, index=text_classes, columns=img_classes)

        fig_heat = px.imshow(
            sim_df,
            labels=dict(x="Image Class", y="Text Prompt Class", color="Cosine Sim"),
            color_continuous_scale="Viridis",
            template="plotly_dark",
            aspect="auto",
        )
        fig_heat.update_layout(
            height=460,
            paper_bgcolor="rgba(18, 24, 36, 0.7)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Feature: Zero-Shot Text-to-Image Latent Search
    st.markdown("### 🔎 Zero-Shot Cross-Modal Text Search")
    st.markdown(
        "Enter any arbitrary natural language query. CLIP will project the query into the 512-D multimodal space "
        "and retrieve the most semantically relevant images based on cross-modal cosine similarity."
    )

    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., Heavy winter jacket with warm fur hood, or comfortable sport sneakers",
            value="Warm knitted sweater for winter",
        )

    if search_query and search_query.strip():
        query_emb = mm_svc.encode_text([search_query.strip()])
        scores = np.dot(query_emb, img_embs.T)[0]
        ranked_indices = np.argsort(scores)[::-1]

        st.markdown(f"**Top Image Matches for:** *\"{search_query.strip()}\"*")
        match_cols = st.columns(5)
        for rank, idx in enumerate(ranked_indices[:5]):
            with match_cols[rank]:
                st.image(images[idx], caption=f"#{rank+1}: {img_classes[idx]} ({scores[idx]:.3f})", width=90)
                st.markdown(
                    f"<div style='font-size: 0.75rem; color: #10b981; font-weight: 600;'>{scores[idx]*100:.1f}% Match</div>",
                    unsafe_allow_html=True,
                )
