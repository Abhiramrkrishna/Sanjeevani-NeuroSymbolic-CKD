# 🏥 Project Sanjeevani: Neuro-Symbolic AI for Automated CKD Triage

**Built for the Kaggle MedGemma Impact Challenge 2025** *By Abhiram Radha Krishna*

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open_Notebook-blue?logo=kaggle)](YOUR_KAGGLE_NOTEBOOK_LINK_HERE)
[![YouTube Video](https://img.shields.io/badge/YouTube-Watch_Demo-red?logo=youtube)](YOUR_YOUTUBE_LINK_HERE)

## 📌 The Problem: A Silent Epidemic
Chronic Kidney Disease (CKD) affects **850 million people worldwide**, yet early stages are entirely asymptomatic. Urinary ¹H NMR spectroscopy can detect early metabolic shifts—depleted citrate, elevated TMAO, elevated lactate—*before* any eGFR decline is measurable. However, NMR interpretation requires expert spectroscopists, creating a critical access bottleneck globally.

**Why Pure LLMs Cannot Solve This:** LLMs are text-prediction engines. They are mathematically incapable of performing spectral deconvolution on raw float arrays and will confidently hallucinate diagnoses. A safe system requires deterministic physics to extract verified biochemical ratios *before* any LLM is invoked.

## ⚙️ The Solution: A 3-Stage Neuro-Symbolic Pipeline
Sanjeevani deploys three AI agents sequentially. Each operates strictly within its domain of competence.

| Stage | Agent | Task | Why this agent |
| --- | --- | --- | --- |
| **1** | **PaliGemma** | **Visual QC:** Classifies 2.5–4.5 ppm ROI as 'sharp peaks' or 'flat noise'; halts pipeline on bad data. | Spatial reasoning on chart image; cannot hallucinate a ratio it never computes. |
| **2** | **SanjeevaniEngine** | **Biomarker Deconvolution:** Custom PyTorch CNN-Transformer extracts verified chemistry. | LLMs cannot process raw float arrays — a deterministic physics engine is required. |
| **3** | **MedGemma** | **Clinical Co-Pilot & EMR Export:** Synthesizes verified ratios into a Physician Assessment and Patient Portal draft. | Clinical language model; receives verified maths only to eliminate hallucination. |

### 🔒 The Dual-Layer Safety Gate
MedGemma **never sees raw spectral data**. It receives only verified decimal ratios produced by the physics engine. Furthermore, PaliGemma's visual QC is backed by a deterministic creatinine biochemical gate, ensuring degraded samples halt the pipeline before false clinical reasoning occurs.

## 🔬 Technical Highlights
* **Independent Decoders:** The PyTorch engine utilizes 5 independent decoder branches with per-channel weighted loss to prevent channel collapse in dense biomarker clusters.
* **Aggressive Triage Tuning:** Achieved **100% Sensitivity** and 89.5% accuracy on valid samples, ensuring no high-risk patients are missed during triage.
* **Edge-Optimized:** The entire 7-Billion parameter multi-agent pipeline is heavily optimized via `bitsandbytes` 4-bit quantization, allowing it to run concurrently on constrained dual-T4 GPUs for on-device patient data privacy.

![Confusion Matrix](assets/con_mat_1.png)

## 📊 Real-World Domain Gap Analysis
To assess clinical readiness, Sanjeevani was evaluated via a zero-shot transfer test on the real-world **MTBLS1 clinical dataset**. This revealed a critical domain gap: real urine spectra exhibit a broad macromolecular baseline floor at 0.3 normalized intensity, compressing target amplitudes. 

![Domain Gap Analysis](assets/spectra.png)

*A model that fails honestly and explicably is more valuable for clinical AI development than one that succeeds silently on its own test set. Future V2 work includes integrating a quantum spin Hamiltonian simulator (Spinach) for physically accurate training spectra.*

## 📂 Repository Contents
* `/code`: Contains the fully reproducible Kaggle notebook.
* `/docs`: Contains the detailed 7-page technical report and the video pitch deck.
* `/assets`: UI screenshots and evaluation plots.
