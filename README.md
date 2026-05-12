<div align="center">

# Uni-SOR

### A unified framework for high-fidelity recovery in spatially-resolved multi-omics and microscopy

**Recover sparse, blurred, low-resolution and degraded spatial signals across omics and imaging modalities with one unified framework.**

<br>

<img src="https://img.shields.io/badge/Spatially--resolved%20multi--omics-Recovery-7C3AED?style=for-the-badge" />
<img src="https://img.shields.io/badge/Microscopy-Restoration-06B6D4?style=for-the-badge" />
<img src="https://img.shields.io/badge/Multi--modal%20Analysis-F59E0B?style=for-the-badge" />
<img src="https://img.shields.io/badge/Demo%20SRP & SIM-10B981?style=for-the-badge" />

<br><br>

**TOF-SIMS 路 DESI-MS 路 MALDI-MS 路 IMC 路 SRT 路 SRP 路 H&E 路 SIM 路 miF 路 IHC**

</div>

---

## Overview

**Uni-SOR** is a unified recovery framework designed for high-fidelity reconstruction of spatially-resolved multi-omics and microscopy data. It targets heterogeneous degradation patterns across spatial assays and imaging systems, including sparse sampling, low spatial resolution, defocus blur, signal loss and modality-specific noise.

The framework is built to recover biologically meaningful spatial structures while preserving local morphology, molecular gradients and cross-modal consistency. Uni-SOR has been validated across a broad spectrum of spatial omics and microscopy modalities, including **TOF-SIMS**, **DESI-MS**, **MALDI-MS**, **IMC**, **SRT**, **SRP**, **H&E**, **SIM**, **miF** and **IHC**.

---

## Why Uni-SOR

Spatially-resolved omics and microscopy platforms often face a trade-off between throughput, field of view, spatial resolution and signal fidelity. Uni-SOR addresses this limitation with a unified recovery strategy that can be adapted to multiple experimental settings.

Uni-SOR supports three recovery scenarios.

- **Sparse-sampling recovery**  
  Reconstruction of high-fidelity spatial signals from sparsely acquired measurements.

- **Super-resolution recovery**  
  Enhancement of spatial detail and local structural continuity from low-resolution inputs.

- **Microscopy restoration**  
  Recovery of microscopy images affected by defocus, degradation or modality-specific signal loss.

---

## Validated modalities

Uni-SOR has been evaluated across spatial omics, mass spectrometry imaging, multiplexed imaging and microscopy modalities.

| Category | Modalities |
| --- | --- |
| Mass spectrometry imaging | TOF-SIMS, DESI-MS, MALDI-MS |
| Spatial omics | SRT, SRP |
| Multiplexed tissue imaging | IMC, miF, IHC |
| Histology and microscopy | H&E, SIM |

These validations show that Uni-SOR can recover high-frequency structures, preserve tissue-level spatial organization and improve signal fidelity across heterogeneous biological data types.

---

## Downstream-ready recovery

Uni-SOR is designed not only for visual restoration, but also for downstream biological and computational analysis. The recovered outputs can support multiple downstream tasks.

- **Cell segmentation with pathology foundation models**  
  Recovered images can be used as inputs for case-level or pathology foundation models to improve cell segmentation in degraded tissue images.

- **IHC prediction from recovered H&E**  
  Recovered H&E images can provide enhanced morphological information for IHC prediction and virtual staining tasks.

- **Joint analysis across MSI, SRT and H&E**  
  Recovered MSI, SRT and H&E data can be integrated for multi-modal spatial analysis, enabling more reliable alignment between molecular signals and tissue morphology.

---

## Repository status

This repository currently provides lightweight demo files for **SRP** and **SIM** tasks. Additional task-specific resources, pretrained weights and full benchmark examples will be released progressively.

For lightweight tasks, we also provide free online access through our website.

> Website link will be added here.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/your-lab/Uni-SOR.git
cd Uni-SOR
```

Create and activate an environment.

```bash
conda create -n unisor python=3.10
conda activate unisor
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Quick start

After downloading the repository, run one of the demo scripts according to the task.

### Sparse-sampling recovery

```bash
python "code/run spare-sampling demo.py"
```

### Super-resolution recovery

```bash
python "code/run super-resolution demo.py"
```

### SIM restoration

```bash
python "code/run SIM demo.py"
```

Before running a demo, modify the file paths and pretrained weight paths in the corresponding script.

```python
input_path = "path/to/your/input"
output_path = "path/to/save/recovered/results"
weight_path = "path/to/pretrained/weights"
```

---

## Demo availability

Only **SRP** and **SIM** demo files are currently included in this repository.

| Demo | Status |
| --- | --- |
| SRP sparse-sampling or super-resolution recovery | Available |
| SIM microscopy restoration | Available |
| TOF-SIMS, DESI-MS, MALDI-MS, IMC, SRT, H&E, miF and IHC demos | Coming soon |

---

## Expected input and output

Uni-SOR takes degraded spatial omics or microscopy data as input and produces recovered high-fidelity outputs.

```text
Input
low-quality spatial signal, sparse measurement, low-resolution image or degraded microscopy image

Output
recovered spatial signal or microscopy image with enhanced fidelity
```

Recovered outputs can be used directly for visualization, quantitative analysis and downstream model-based tasks.

---

## Framework

```text
Degraded spatial data
        鈹�
        鈻�
Unified recovery backbone
        鈹�
        鈹溾攢鈹€ sparse-sampling recovery
        鈹溾攢鈹€ super-resolution recovery
        鈹斺攢鈹€ microscopy restoration
        鈹�
        鈻�
High-fidelity recovered output
        鈹�
        鈹溾攢鈹€ cell segmentation
        鈹溾攢鈹€ IHC prediction
        鈹斺攢鈹€ MSI-SRT-H&E joint analysis
```

---

## Applications

Uni-SOR is designed for research scenarios where spatial fidelity directly affects biological interpretation.

- Recovery of sparsely sampled spatial proteomics data
- Enhancement of low-resolution spatial transcriptomics data
- Restoration of microscopy images with degraded visual quality
- Cross-modal integration of MSI, SRT and histology
- Preprocessing for foundation-model-based tissue analysis
- Improved visualization of spatial molecular patterns

---

## Citation

If Uni-SOR is useful for your research, please cite our work.

```bibtex
@article{unisor2026,
  title={Uni-SOR: A unified framework for high-fidelity recovery in spatially-resolved multi-omics and microscopy},
  author={Your Name and Collaborators},
  journal={},
  year={2026}
}
```

---

## License

This project is released for academic research use. Please check the license file for details.

---

<div align="center">

**Uni-SOR**  
**Unified spatial omics and microscopy recovery for high-fidelity biological discovery**

</div>
