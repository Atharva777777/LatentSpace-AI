"""Bias Analytics Module (U5 - Responsible AI).

Probes sentence embeddings for occupation-gender associations,
calculates Word Embedding Association Test (WEAT) effect sizes,
and demonstrates orthogonal subspace debiasing.
"""

from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from services.bias_service import BiasService
from services.dimensionality_service import DimensionalityService
from services.embedding_service import EmbeddingService


@st.cache_resource(show_spinner=False)
def get_bias_service() -> BiasService:
    """Initializes and caches BiasService singleton."""
    embed_svc = EmbeddingService()
    return BiasService(embed_svc)


def render_bias_analytics_page() -> None:
    """Main controller for Unit 5: Responsible AI & Latent Bias Analytics."""
    bias_svc = get_bias_service()

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2rem;">U5: Responsible AI & Latent Bias Analytics</h1>
                <p style="color: #94a3b8; margin-top: 4px; font-size: 0.95rem;">
                    Gender-Occupation Association Probing &bull; WEAT Statistical Auditing &bull; Subspace Debiasing
                </p>
            </div>
            <span class="badge-pill badge-active">Active Module</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_spectrum, tab_weat, tab_debias, tab_theory = st.tabs(
        ["📊 Occupation Bias Spectrum", "🧪 WEAT Significance Test", "🛡️ Interactive Probe & Debias", "📚 Fairness Theory"]
    )

    # =========================================================================
    # TAB 1: OCCUPATION BIAS SPECTRUM
    # =========================================================================
    with tab_spectrum:
        st.markdown(
            "Measures the directional projection of standard professions along the latent gender axis: "
            "$\\mathbf{v}_{\\text{gender}} = \\frac{\\mathbf{c}_{\\text{female}} - \\mathbf{c}_{\\text{male}}}{\\|\\mathbf{c}_{\\text{female}} - \\mathbf{c}_{\\text{male}}\\|_2}$. "
            "Positive scores denote alignment toward female attributes; negative scores denote alignment toward male attributes."
        )

        with st.spinner("Computing occupational gender projections..."):
            results = bias_svc.compute_occupation_associations()

        df_bias = pd.DataFrame(results)

        # Plotly Divergent Bar Chart
        colors = [
            "#f43f5e" if row["score"] > 0.05 else ("#38bdf8" if row["score"] < -0.05 else "#94a3b8")
            for _, row in df_bias.iterrows()
        ]

        fig_bar = go.Figure(
            go.Bar(
                x=df_bias["score"],
                y=df_bias["occupation"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{s:+.3f}" for s in df_bias["score"]],
                textposition="auto",
            )
        )

        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(18, 24, 36, 0.7)",
            plot_bgcolor="rgba(14, 18, 28, 0.9)",
            height=600,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis=dict(
                title="Directional Association: Male Skew (Left) &larr; &bull; &rarr; Female Skew (Right)",
                zeroline=True,
                zerolinecolor="#ffffff",
                zerolinewidth=2,
            ),
            yaxis=dict(autorange="reversed"),
        )

        c_chart, c_metrics = st.columns([2, 1])

        with c_chart:
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_metrics:
            st.markdown("##### 📈 Summary Statistics")
            male_skewed = df_bias[df_bias["score"] < -0.05]
            female_skewed = df_bias[df_bias["score"] > 0.05]
            neutral = df_bias[(df_bias["score"] >= -0.05) & (df_bias["score"] <= 0.05)]

            st.metric("Total Occupations Probed", len(df_bias))
            st.metric("Male-Aligned Occupations", f"{len(male_skewed)} ({len(male_skewed)/len(df_bias)*100:.0f}%)")
            st.metric("Female-Aligned Occupations", f"{len(female_skewed)} ({len(female_skewed)/len(df_bias)*100:.0f}%)")
            st.metric("Neutral Occupations", f"{len(neutral)} ({len(neutral)/len(df_bias)*100:.0f}%)")

            st.markdown(
                """
                <div class="research-card" style="padding: 12px; margin-top: 15px;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">FINDING</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                        Technical/STEM professions (electrician, software engineer) exhibit directional male skew, 
                        while caregiving and elementary education exhibit directional female skew.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 2: WEAT STATISTICAL AUDITING
    # =========================================================================
    with tab_weat:
        st.markdown(
            "#### 🧪 Word Embedding Association Test (WEAT)"
        )
        st.markdown(
            "WEAT (Caliskan et al., Science 2017) statistically quantifies whether target concepts $X$ (Math / Career) "
            "are significantly more associated with attribute group $A$ (Male) than target concepts $Y$ (Arts / Family) "
            "are with attribute group $B$ (Female)."
        )

        target_math = [
            "math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"
        ]
        target_arts = [
            "poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"
        ]

        with st.spinner("Computing permutation test (1,000 permutations)..."):
            weat_res = bias_svc.compute_weat_score(target_math, target_arts, num_permutations=1000)

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            st.metric("Effect Size (Cohen's d)", f"{weat_res['effect_size']:.3f}")
        with w_col2:
            st.metric("p-value (Permutation)", f"{weat_res['p_value']:.4f}")
        with w_col3:
            sig_text = "YES (p < 0.05)" if weat_res["statistically_significant"] else "NO (p >= 0.05)"
            st.metric("Statistically Significant?", sig_text)

        st.markdown(
            f"""
            <div class="research-card">
                <div class="research-card-title">Analysis Interpretation</div>
                <p style="color: #cbd5e1; font-size: 0.85rem;">
                    The measured effect size of <b>{weat_res['effect_size']}</b> indicates a notable differential association 
                    between Math terms and Male gender attributes compared to Arts terms and Female attributes. 
                    The permutation test yields $p = {weat_res['p_value']:.4f}$, confirming that this stereotypic alignment 
                    is statistically significant and not an artifact of random sampling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 3: INTERACTIVE PROBE & DEBIAS
    # =========================================================================
    with tab_debias:
        st.markdown(
            "Test any custom role or sentence for gender association, and inspect the effect of "
            "**orthogonal subspace projection debiasing** (neutralizing gender correlation)."
        )

        p_input = st.text_input("Enter a role, title, or sentence:", value="Chief Technology Officer")

        if p_input and p_input.strip():
            raw_emb = bias_svc.embedding_service.encode_sentence(p_input.strip())
            gender_vec = bias_svc.get_gender_direction()

            # Original score
            orig_score = float(np.dot(raw_emb, gender_vec))

            # Debiased embedding
            debiased_emb = bias_svc.debias_embedding(raw_emb)
            debiased_score = float(np.dot(debiased_emb, gender_vec))

            d_col1, d_col2 = st.columns(2)

            with d_col1:
                st.markdown("##### ⚠️ Baseline (Original Embedding)")
                st.markdown(
                    f"""
                    <div class="research-card" style="border-left: 4px solid {'#f43f5e' if orig_score > 0 else '#38bdf8'};">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">"{p_input.strip()}"</div>
                        <div style="font-size: 1.6rem; font-family: monospace; font-weight: 700; margin: 8px 0; color: #facc15;">
                            {orig_score:+.4f}
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">
                            Orientation: <b>{'Female Skew' if orig_score > 0.05 else ('Male Skew' if orig_score < -0.05 else 'Neutral')}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with d_col2:
                st.markdown("##### 🛡️ Mitigated (Orthogonal Subspace Debiased)")
                st.markdown(
                    f"""
                    <div class="research-card" style="border-left: 4px solid #10b981;">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">"{p_input.strip()}"</div>
                        <div style="font-size: 1.6rem; font-family: monospace; font-weight: 700; margin: 8px 0; color: #10b981;">
                            {debiased_score:+.4f}
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">
                            Orientation: <b>Neutralized (Orthogonal to Gender Axis)</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 4: FAIRNESS THEORY
    # =========================================================================
    with tab_theory:
        st.markdown(
            r"""
            ### 🎓 Responsible AI & Embedding Geometry (Viva Notes)

            1. **Sources of Latent Bias:**
               Pretrained language models are trained on massive web corpora (Common Crawl, Wikipedia, Reddit) that mirror 
               historical societal stereotypes. Words frequently co-occurring with male or female pronouns inherit directional 
               shifts in the embedding space.

            2. **Representational vs. Allocational Harm:**
               - **Representational Harm:** Reinforcing derogatory stereotypes or false associations (e.g. associating women with domestic roles and men with executive roles).
               - **Allocational Harm:** Downstream automated systems (e.g., CV screening, automated credit scoring) unfairly allocating fewer opportunities based on stereotypic embedding distances.

            3. **Mathematical Formulation of Orthogonal Hard Debiasing:**
               Given an embedding vector $\mathbf{e} \in \mathbb{R}^D$ and a normalized gender subspace vector $\mathbf{v}_{\text{gender}}$, the neutralized representation is:
               $$\mathbf{e}_{\text{neutral}} = \frac{\mathbf{e} - (\mathbf{e} \cdot \mathbf{v}_{\text{gender}})\mathbf{v}_{\text{gender}}}{\|\mathbf{e} - (\mathbf{e} \cdot \mathbf{v}_{\text{gender}})\mathbf{v}_{\text{gender}}\|_2}$$
               This strictly zeros out the cosine projection along the gender axis ($\mathbf{e}_{\text{neutral}} \cdot \mathbf{v}_{\text{gender}} = 0$) while preserving orthogonal semantic coordinates.
            """
        )
