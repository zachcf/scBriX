# RNA-bridged cross-modal generation from partially paired single-cell measurements


---

## About The Project

Single-cell multimodal profiling links complementary aspects of cellular identity and function, but joint assays capture only selected combinations of measurements. Here we present scBriX, an RNA-bridged diffusion framework that reuses paired RNA–modality measurements from separate cell populations to generate features between modalities lacking direct paired training data.  A fixed RNA reference coordinates alignment, while hierarchical constraints are designed to retain modality-specific structure. At inference, scBriX generates target features without measured or intermediate RNA. We demonstrate bidirectional generation between chromatin accessibility and surface-protein abundance, generation of T cell receptor–derived features, and prediction of neuronal morphological and electrophysiological features from chromatin accessibility. New modalities are incorporated by training their associated modules while keeping existing modules fixed. scBriX thus enables the reuse and extension of RNA-linked supervision for cross-modal generation beyond directly co-measured modality combinations.

---
![Alt text](./data/scBriDi.png?raw=true "scBriDi")


## Built With

- Python 3.11.7
- PyTorch 2.3.1
- scanpy / anndata / episcanpy
- scikit-learn / scipy / numpy
- einops / timm
- tqdm

---

## Getting Started

```

### Installation

```bash
pip install torch scanpy anndata episcanpy scikit-learn scipy numpy einops timm tqdm
```

---




### Tutorials

| Notebook | Description |
|----------|-------------|
| `tutorial/BridgingAlignment.ipynb` | Two-stage bridging alignment (RNA–ATAC → RNA–ADT) with cross-modal generation |
| `tutorial/CoordinateAssignment.ipynb` | Spatial coordinate assignment for scRNA-seq cells |

---


## License

Distributed under the MIT License.

---

## Citation

If you find scBriDi useful, please cite:

> scBriDi: an RNA-centered bridging alignment and cross-modal generative framework for single-cell multi-omics

---

## Acknowledgments

- [scanpy](https://scanpy.readthedocs.io/), [anndata](https://anndata.readthedocs.io/), and [episcanpy](https://github.com/colomemaria/epiScanpy) for single-cell data handling
