# Machine Learning for Physics — Project Portfolio

Three deep learning projects on real and simulated physics data, completed as the final project of a graduate-level **Data Science for Physics Graduates** course (Computational Physics I & II) at **Ben-Gurion University of the Negev**, 2026.

Each notebook is self-contained, with the reasoning, code, plots and results written out step by step.

**Tools:** Python · PyTorch · PyTorch Geometric · scikit-learn · NumPy · Pandas · SciPy · Matplotlib

---

## 1. Photometric Redshift Estimation under Color–Redshift Degeneracy
📓 [`1_photometric_redshifts_mdn.ipynb`](1_photometric_redshifts_mdn.ipynb)

Quasars at very different redshifts can have nearly identical colors, so the true posterior p(z | colors) is often **multi-modal**, and a single point estimate with an error bar is misleading.

- Used real **SDSS quasar** photometry (ugriz colors → spectroscopic redshift)
- Confirmed the degeneracy with a training-free **nearest-neighbors** check (about 1 in 5 quasars has a clearly bimodal redshift neighborhood)
- Built a **Gaussian mean–variance network** and a **Mixture Density Network (MDN)**, both trained with negative log-likelihood
- Prevented variance collapse with a variance floor, weight decay, early stopping and learning-rate scheduling
- Evaluated calibration with **coverage curves** and the **probability integral transform (PIT)**

**Result:** the MDN recovers both redshift solutions for degenerate quasars and is better calibrated, while the Gaussian baseline can only place one wide peak between them.

---

## 2. Molecular Property Prediction with GNNs and Transformers (QM9)
📓 [`2_molecular_property_gnn_qm9.ipynb`](2_molecular_property_gnn_qm9.ipynb)

Predicting the **HOMO–LUMO energy gap** of ~130k small molecules from their 3D structure.

- **Baseline:** MLP on atom-type counts only (no structure)
- **Message-passing GNN** (PyTorch Geometric) with interatomic distances as rotation/translation-invariant edge features and mean pooling
- **Set transformer** with self-attention over atoms and padding masks
- Verified **permutation invariance** automatically (shuffled atom order → identical predictions)
- Uncertainty from **deep ensembles** and **bootstrap** resampling; evaluated calibration, error–uncertainty correlation and **out-of-distribution** detection

| Model | Test MAE (eV) |
|---|---|
| Atom-count baseline | 0.839 |
| GNN (ensemble) | 0.296 |
| Transformer (ensemble) | 0.222 |

---

## 3. Generative Modeling of the Ising Model with Diffusion
📓 [`3_ising_diffusion_model.ipynb`](3_ising_diffusion_model.ipynb)

Can a generative model learn to produce physically correct spin configurations?

- Generated training data with **Metropolis Monte Carlo** simulation of the 2D Ising model
- Implemented a **denoising diffusion model** from scratch: forward noising, noise-prediction training, reverse sampling
- Compared a **CNN denoiser** (circular padding, local structure) with an **MLP denoiser**
- Extended it to a **temperature-conditional** generator across the phase transition (T from 1.0 to 3.0, through T_c ≈ 2.269)
- Validated generated samples against physical observables: magnetization and energy per spin

**Result:** the CNN generates coherent ordered domains within physical bounds and captures the shape of the phase transition; the MLP, even with a larger network, does not.

---

### Notes
- Datasets are downloaded inside the notebooks (SDSS, QM9 via PyTorch Geometric). Data files and trained model weights are not included in this repository because of their size.

**Author:** Eden Koter · kotereden@gmail.com
