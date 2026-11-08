# LATENTSPACE AI: Multimodal Intelligence Explorer

> **Generative AI CA Mini Project &bull; Problem Statement P6**  
> An autonomous research application for exploring, projecting, generating, and analyzing semantic latent spaces across text, vision, and multimodal representations.

---

## 1. Project Overview & Curriculum Alignment

**Assigned Problem Statement (P6): Text & Image Latent-Space Explorer**  
*Goal:* Visualise how natural language sentences and images are placed in a shared continuous latent space, inspect nearest-neighbour topologies, train generative models, and probe representations for bias.

All five mandatory units are **fully implemented, tested, and operational**:

* **U1 &ndash; Foundations:** Projects high-dimensional sentence embeddings (384-D) to 2D with PCA and UMAP; interactive Plotly scatter plot with point selection; exact high-dimensional cosine similarity search; dynamic corpus additions; viva guide.
* **U2 &ndash; GANs & LLMs:** Partitions embeddings into semantic clusters; prompts an instruction-tuned Seq2Seq LLM (`google/flan-t5-small`) to auto-name each cluster from centroid exemplars; trains a convolutional Variational Autoencoder (VAE) on Fashion-MNIST with Evidence Lower Bound (ELBO) loss.
* **U3 &ndash; GPT, BERT & Multimodal:** Jointly embeds natural language text and Fashion-MNIST images into a unified 512-D multimodal space using OpenAI CLIP (`clip-ViT-B-32`); co-locates text and images in a shared 2D projection with alignment vectors; cross-modal similarity heatmaps; zero-shot multimodal search.
* **U4 &ndash; Generative AI Tools:** Interactive point-and-decode continuous latent synthesizer using the trained Fashion-MNIST VAE decoder $g_\theta(\mathbf{z})$; geodesic latent interpolation between fashion prototypes; full continuous 2D manifold canvas rendering.
* **U5 &ndash; Responsible AI:** Quantifies directional projections of professions onto the latent gender axis; computes Word Embedding Association Test (WEAT) effect sizes ($d$) with permutation $p$-values; implements orthogonal subspace hard debiasing ($\mathbf{e}_{\text{neutral}} \cdot \mathbf{v}_{\text{gender}} = 0$).

---

## 2. Technology Stack

* **Language:** Python 3.11+ (Validated on Python 3.12.10)
* **Frontend & Dashboard:** [Streamlit](https://streamlit.io/) (Custom dark research theme in `assets/style.css`)
* **Pretrained Models:**
  * Text Embeddings: [Sentence-Transformers](https://sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`, 384-D)
  * Multimodal Embeddings: [CLIP](https://huggingface.co/sentence-transformers/clip-ViT-B-32) (`clip-ViT-B-32`, 512-D)
  * LLM Cluster Namer: [Flan-T5](https://huggingface.co/google/flan-t5-small) (`google/flan-t5-small` Seq2Seq)
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/) (CPU-compatible)
* **Dimensionality Reduction:**
  * PCA: [Scikit-learn](https://scikit-learn.org/) (`sklearn.decomposition.PCA`)
  * UMAP: [UMAP-learn](https://umap-learn.readthedocs.io/) (`umap.UMAP`)
* **Interactive Visualisation:** [Plotly Graph Objects](https://plotly.com/python/) & Plotly Express
* **Data Processing:** Pandas, NumPy, Pillow
* **Automated Testing:** Pytest

---

## 3. Project Architecture

The application enforces strict separation of concerns across presentation, business logic, ML inference, and persistent data:

```
TEXTLE/
├── app.py                      # Main entry point and sidebar module router
├── requirements.txt            # Production dependencies
├── README.md                   # Comprehensive documentation & viva guide
├── .gitignore                  # Git ignore rules for venv, cache, and large weights
│
├── modules/                    # UI presentation & page controllers
│   ├── __init__.py
│   ├── embedding_explorer.py   # [U1] 2D latent projection & 384-D cosine search
│   ├── cluster_intelligence.py # [U2] K-Means clustering, Flan-T5 auto-naming, VAE training
│   ├── multimodal_explorer.py  # [U3] Shared CLIP text + image 512-D manifold & zero-shot search
│   ├── generative_studio.py    # [U4] Continuous VAE point-and-decode & latent interpolation
│   └── bias_analytics.py       # [U5] Gender-occupation bias spectrum, WEAT & debiasing
│
├── services/                   # Decoupled AI/ML services
│   ├── __init__.py
│   ├── embedding_service.py    # all-MiniLM-L6-v2 inference with in-memory hash cache
│   ├── dimensionality_service.py # PCA & UMAP mathematical reduction engine
│   ├── similarity_service.py   # Exact 384-D cosine nearest-neighbour search
│   ├── clustering_service.py   # K-Means clustering & centroid exemplar extraction
│   ├── llm_naming_service.py   # Prompt synthesis & Seq2Seq LLM cluster naming
│   ├── vae_service.py          # PyTorch Fashion-MNIST VAE (encoder, decoder, ELBO loss)
│   ├── multimodal_service.py   # CLIP text & image encoding, cross-modal similarity
│   └── bias_service.py         # Gender subspace probing, WEAT metrics & orthogonal debias
│
├── data/
│   ├── sample_sentences.csv    # 42 balanced sentences across 7 semantic categories
│   └── fashion_mnist/          # Fashion-MNIST dataset files
│
├── models/
│   ├── all-MiniLM-L6-v2/       # Local Sentence-BERT weights for 100% offline execution
│   ├── clip-ViT-B-32/          # Local CLIP weights for offline multimodal execution
│   └── fashion_mnist_vae.pt    # Pre-trained VAE checkpoint (reused across U2 & U4)
│
├── assets/
│   └── style.css               # Modern dark research dashboard CSS styling
│
└── tests/
    ├── test_embeddings.py      # U1 unit tests (9 tests)
    ├── test_integration.py     # U1 end-to-end integration tests (4 tests)
    ├── test_u2.py              # U2 clustering, LLM naming & VAE architecture tests (4 tests)
    ├── test_u3.py              # U3 CLIP text, image & shared projection tests (4 tests)
    ├── test_u4.py              # U4 VAE latent decoding, interpolation & grid tests (3 tests)
    └── test_u5.py              # U5 gender subspace, WEAT & debiasing tests (4 tests)
```

---

## 4. Implementation Details by Unit

### Unit 1: Foundations (Embedding Explorer)
* **Text Dataset:** 42 curated sentences across 7 categories (`Animals`, `Technology`, `Food`, `Transportation`, `Sports`, `Education`, and `Nature`).
* **High-D Embeddings:** `all-MiniLM-L6-v2` produces dense 384-D vectors with hash-based caching.
* **PCA & UMAP:** Switchable dimensionality reduction with variance metrics and tunable hyperparameters.
* **Exact Latent Cosine Search:** Evaluated directly in the uncompressed 384-dimensional space (not in 2D projected space).
* **Dynamic Additions:** User-added sentences immediately re-project and shift active focus.

### Unit 2: GANs & LLMs (Cluster Intelligence & VAE Training)
* **Semantic Clustering:** K-Means clustering in 384-D space with cosine silhouette evaluation.
* **LLM Auto-Naming:** Generates structured prompts with centroid exemplars and c-TF-IDF keywords; queries `google/flan-t5-small` to produce concise topic labels.
* **Fashion-MNIST VAE:** Convolutional VAE with a 2D continuous latent space $\mathbf{z} \in \mathbb{R}^2$. Trains with Adam on CPU in ~15 seconds using ELBO loss:
  $$\mathcal{L}_{\text{VAE}} = \text{BCE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q(z|x) \parallel p(z))$$
* **Checkpoint Reuse:** Saves model weights to `models/fashion_mnist_vae.pt` for direct consumption in Unit 4.

### Unit 3: GPT, BERT & Multimodal (Multimodal Explorer)
* **Joint Embedding:** Natural language descriptions and Fashion-MNIST apparel images encoded into a unified 512-D space via `clip-ViT-B-32`.
* **Shared 2D Manifold:** Projects text (circles) and images (diamonds) into a single coordinate frame with dotted semantic correspondence vectors.
* **Cross-Modal Heatmap:** Pairwise cosine similarity matrix demonstrating semantic alignment between descriptions and apparel classes.
* **Zero-Shot Multimodal Search:** Free-form text query retrieves closest images via cross-modal cosine similarity.

### Unit 4: Generative AI Tools (Generative Studio)
* **Point-and-Decode Synthesis:** Interactive 2D coordinate sliders $(z_1, z_2)$ and empirical latent scatter plot; decoder network $g_\theta(\mathbf{z})$ synthesizes a $28 \times 28$ image in real time.
* **Geodesic Interpolation:** Linear interpolation $\mathbf{z}(t) = (1 - t)\mathbf{z}_A + t\mathbf{z}_B$ across latent fashion prototypes (e.g., Sneaker $\to$ Ankle boot) showing smooth morphing.
* **2D Manifold Canvas:** Decodes a regular grid (e.g. $8 \times 8$) spanning $[-2.5, 2.5] \times [-2.5, 2.5]$ showing the full learned topology.

### Unit 5: Responsible AI (Bias Analytics)
* **Gender Subspace Probing:** Unit vector $\mathbf{v}_{\text{gender}} = \frac{\mathbf{c}_{\text{female}} - \mathbf{c}_{\text{male}}}{\|\mathbf{c}_{\text{female}} - \mathbf{c}_{\text{male}}\|_2}$. Measures directional projection $s(w) = \cos(\mathbf{e}_w, \mathbf{v}_{\text{gender}})$ across 20+ occupations.
* **WEAT Auditing:** Benchmark evaluation (Math vs Arts against Male vs Female attributes) reporting Cohen's $d$ effect size and permutation test $p$-value ($p < 0.05$).
* **Orthogonal Subspace Debiasing:** Interactive neutralization:
  $$\mathbf{e}_{\text{neutral}} = \frac{\mathbf{e} - (\mathbf{e} \cdot \mathbf{v}_{\text{gender}})\mathbf{v}_{\text{gender}}}{\|\mathbf{e} - (\mathbf{e} \cdot \mathbf{v}_{\text{gender}})\mathbf{v}_{\text{gender}}\|_2}$$
  Strictly zeros out gender correlation while preserving orthogonal semantic meaning.

---

## 5. Execution & Verification

### Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Running the Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. Use the sidebar to navigate between U1, U2, U3, U4, and U5.

### Running the Automated Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/
```

**28 passing tests across all units:**
* `tests/test_embeddings.py` (9 tests passed) &ndash; Embeddings, PCA, UMAP, cosine similarity, nearest neighbours.
* `tests/test_integration.py` (4 tests passed) &ndash; Full U1 pipeline, dataset integrity, dynamic additions.
* `tests/test_u2.py` (4 tests passed) &ndash; K-Means clustering, exemplar extraction, LLM naming, VAE architecture.
* `tests/test_u3.py` (4 tests passed) &ndash; CLIP text and image encoding, joint 2D projection, cross-modal similarity.
* `tests/test_u4.py` (3 tests passed) &ndash; VAE coordinate decoding, interpolation, manifold grid generation.
* `tests/test_u5.py` (4 tests passed) &ndash; Gender subspace norm, occupation projections, WEAT metric, orthogonal debiasing.
