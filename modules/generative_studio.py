"""Generative Studio Module (U4 - Generative AI Tools).

Interactive continuous latent decoding and traversal using the trained
Fashion-MNIST Variational Autoencoder (VAE).
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from services.vae_service import VAEService

CLASS_COLORS = [
    "#38bdf8", "#818cf8", "#c084fc", "#f43f5e", "#f59e0b",
    "#10b981", "#06b6d4", "#e11d48", "#84cc16", "#d946ef",
]


@st.cache_resource(show_spinner=False)
def get_vae_service() -> VAEService:
    """Initializes and caches VAEService with loaded weights."""
    svc = VAEService(latent_dim=2)
    svc.load_checkpoint()
    return svc


@st.cache_data(show_spinner=False)
def compute_dataset_latent_projections(
    num_samples: int = 400,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Encodes a sample of Fashion-MNIST test images into the 2D latent space."""
    svc = get_vae_service()
    transform = transforms.ToTensor()
    data_dir = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"
    test_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=False, transform=transform
    )

    indices = list(range(min(num_samples, len(test_dataset))))
    images = []
    labels = []
    class_names = []

    for i in indices:
        img, lbl = test_dataset[i]
        images.append(img)
        labels.append(lbl)
        class_names.append(svc.FASHION_CLASSES[lbl])

    batch = torch.stack(images).to(svc.device)
    with torch.no_grad():
        mu, _ = svc.model.encode(batch)
        z_coords = mu.cpu().numpy()

    return z_coords, class_names, labels


@st.cache_data(show_spinner=False)
def generate_manifold_grid(grid_size: int = 8, z_range: float = 2.5) -> Image.Image:
    """Decodes a regular 2D grid in latent space into a single composite PIL image."""
    svc = get_vae_service()
    z1_vals = np.linspace(-z_range, z_range, grid_size)
    z2_vals = np.linspace(-z_range, z_range, grid_size)

    canvas = np.zeros((grid_size * 28, grid_size * 28), dtype=np.uint8)

    for i, z2 in enumerate(reversed(z2_vals)):  # Reverse so +z2 is top
        for j, z1 in enumerate(z1_vals):
            img_patch = svc.decode_coordinate(float(z1), float(z2))
            patch_uint8 = (img_patch * 255).clip(0, 255).astype(np.uint8)
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = patch_uint8

    return Image.fromarray(canvas, mode="L")


def render_generative_studio_page() -> None:
    """Main controller for Unit 4: Generative VAE Studio."""
    svc = get_vae_service()

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">U4: Generative VAE Latent Studio</h1>
                <p style="color: #94a3b8; margin-top: 4px; font-size: 0.95rem;">
                    Continuous Latent Space Point-and-Decode Synthesis &bull; Manifold Traversal & Interpolation
                </p>
            </div>
            <span class="badge-pill badge-active">Active Module</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not svc.is_trained():
        st.warning(
            "⚠️ No trained VAE checkpoint found in `models/fashion_mnist_vae.pt`. "
            "Please visit **U2: Cluster Intelligence** to train the VAE or trigger automatic training below."
        )
        if st.button("Train VAE Now (4 Epochs)"):
            with st.spinner("Training Fashion-MNIST VAE..."):
                svc.train_vae(epochs=4, subset_size=2000)
                st.success("Trained successfully!")
                st.rerun()
        return

    tab_decode, tab_interp, tab_manifold = st.tabs(
        ["🎯 Point-and-Decode Synthesizer", "🔄 Latent Morphing / Interpolation", "🗺️ Complete 2D Manifold Grid"]
    )

    # =========================================================================
    # TAB 1: POINT-AND-DECODE
    # =========================================================================
    with tab_decode:
        st.markdown(
            "Select any continuous 2D coordinate $\\mathbf{z} = (z_1, z_2) \\in \\mathbb{R}^2$ on the manifold. "
            "The VAE decoder $g_\\theta(\\mathbf{z})$ immediately reconstructs a novel $28 \\times 28$ fashion garment."
        )

        try:
            z_points, class_names, _ = compute_dataset_latent_projections(num_samples=300)
        except Exception as e:
            st.error(f"Failed to project test samples: {e}")
            return

        col_left, col_right = st.columns([1.5, 1])

        with col_right:
            st.markdown("##### 🎛️ Latent Vector Controls")
            z1 = st.slider("Latent Dimension $z_1$", min_value=-3.5, max_value=3.5, value=0.0, step=0.1)
            z2 = st.slider("Latent Dimension $z_2$", min_value=-3.5, max_value=3.5, value=0.0, step=0.1)

            # Synthesize image at (z1, z2)
            decoded_img = svc.decode_coordinate(z1, z2)
            upscaled = (
                Image.fromarray((decoded_img * 255).astype(np.uint8), mode="L")
                .resize((196, 196), Image.Resampling.NEAREST)
            )

            st.markdown("##### 🖼️ Synthesized Latent Output")
            st.image(
                upscaled,
                caption=f"Decoded from coordinate z = ({z1:.2f}, {z2:.2f})",
                width=196,
            )

            st.markdown(
                f"""
                <div class="research-card" style="padding: 10px 14px; margin-top: 10px;">
                    <div style="font-size: 0.75rem; color: #94a3b8;"><b>Latent Vector:</b> <span class="code-inline">[{z1:.3f}, {z2:.3f}]</span></div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;"><b>Prior Density:</b> $p(\\mathbf{{z}}) = \\mathcal{{N}}(0, I)$</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_left:
            st.markdown("##### 🌌 Empirical 2D Latent Distribution")

            # Build scatter plot of actual Fashion-MNIST test points in VAE latent space
            df_latent = pd.DataFrame({
                "z1": z_points[:, 0],
                "z2": z_points[:, 1],
                "class": class_names,
            })

            fig_latent = px.scatter(
                df_latent,
                x="z1",
                y="z2",
                color="class",
                template="plotly_dark",
                color_discrete_sequence=CLASS_COLORS,
                hover_data=["class"],
            )

            # Highlight currently selected (z1, z2) coordinate
            fig_latent.add_trace(
                go.Scatter(
                    x=[z1],
                    y=[z2],
                    mode="markers",
                    name="Target Point z",
                    marker=dict(size=20, symbol="cross", color="#facc15", line=dict(width=2, color="#ffffff")),
                )
            )

            fig_latent.update_layout(
                paper_bgcolor="rgba(18, 24, 36, 0.7)",
                plot_bgcolor="rgba(14, 18, 28, 0.9)",
                height=480,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(range=[-4, 4], title="Latent z1"),
                yaxis=dict(range=[-4, 4], title="Latent z2"),
                legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_latent, use_container_width=True)

    # =========================================================================
    # TAB 2: LATENT INTERPOLATION
    # =========================================================================
    with tab_interp:
        st.markdown(
            "Interpolate smoothly along a geodesic line between two prototype points in latent space: "
            "$\\mathbf{z}(t) = (1 - t)\\mathbf{z}_A + t\\mathbf{z}_B$, demonstrating the continuous semantic topology."
        )

        i_c1, i_c2, i_c3 = st.columns([1, 1, 1])
        with i_c1:
            proto_a = st.selectbox("Prototype A (Start)", options=svc.FASHION_CLASSES, index=7)  # Sneaker
        with i_c2:
            proto_b = st.selectbox("Prototype B (End)", options=svc.FASHION_CLASSES, index=9)  # Ankle boot
        with i_c3:
            steps = st.slider("Interpolation Steps", min_value=5, max_value=9, value=7)

        # Estimate class centers in latent space
        z_points, class_names, _ = compute_dataset_latent_projections(num_samples=300)
        class_arr = np.array(class_names)
        matches_a = z_points[class_arr == proto_a]
        center_a = matches_a.mean(axis=0) if len(matches_a) > 0 else np.array([0.0, 0.0])
        matches_b = z_points[class_arr == proto_b]
        center_b = matches_b.mean(axis=0) if len(matches_b) > 0 else np.array([1.0, 1.0])

        # Compute interpolated points
        alphas = np.linspace(0.0, 1.0, steps)
        interp_cols = st.columns(steps)

        for step_idx, alpha in enumerate(alphas):
            z_interp = (1.0 - alpha) * center_a + alpha * center_b
            dec_patch = svc.decode_coordinate(float(z_interp[0]), float(z_interp[1]))
            patch_img = (
                Image.fromarray((dec_patch * 255).clip(0, 255).astype(np.uint8), mode="L")
                .resize((100, 100), Image.Resampling.NEAREST)
            )

            with interp_cols[step_idx]:
                st.image(
                    patch_img,
                    caption=f"t = {alpha:.2f}",
                    width=90,
                )

    # =========================================================================
    # TAB 3: MANIFOLD GRID
    # =========================================================================
    with tab_manifold:
        st.markdown(
            "Examine the complete continuous generative manifold learned by the VAE across a regular 2D grid: "
            "$\\mathbf{z} \\in [-2.5, 2.5] \\times [-2.5, 2.5]$."
        )

        m_col1, m_col2 = st.columns([1, 3])
        with m_col1:
            grid_dim = st.select_slider("Grid Resolution", options=[6, 8, 10], value=8)
            z_span = st.slider("Latent Span", min_value=1.5, max_value=3.5, value=2.5, step=0.5)

        with m_col2:
            manifold_img = generate_manifold_grid(grid_size=grid_dim, z_range=z_span)
            st.image(
                manifold_img.resize((500, 500), Image.Resampling.NEAREST),
                caption=f"Continuous 2D Generative Manifold ({grid_dim}x{grid_dim} Grid)",
                width=500,
            )
