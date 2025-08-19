# Physically Plausible Data Augmentations for Wearable IMU-based HAR

This repository contains the implementation of **Physically Plausible Data Augmentation (PPDA) for wearable Inertial Measurement Unit (IMU)-based Human Activity Recognition (HAR)**

[📄 Read our paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)  

---

![Comparison of Physically Plausible Data Augmentations (PPDA) and Signal Transformation-based Data Augmentations (STDA)](assets/ppda-vs-stda.jpg)

*Figure: PPDAs introduce physically plausible variations in wearable IMU data using physics simulation, compared to conventional STDAs which directly apply signal transformations.*

## 📌 Introduction

This repository provides the official implementation of **Physically Plausible Data Augmentation (PPDA)** for **wearable IMU-based Human Activity Recognition (HAR)**. PPDA uses the WIMUSim physics simulation framework to generate realistic variations in:

- **Movement Amplitude**
- **Movement Speed**
- **Sensor Placement**
- **Hardware-related Effects** (noise and bias)

These augmentations help improve model generalization and reduce the need for large-scale labeled data collection, compared to traditional **Signal Transformation-based Data Augmentations (STDAs)**.


## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- Git

1. **Clone this repository and WIMUSim repository**  
```bash
git clone https://github.com/USERNAME/PPDA.git
git clone https://github.com/STRCWearlab/WIMUSim.git
cd PPDA
```

2. Install dependencies
```bash
% 
pip install -e WIMUSim
pip install -r requirements.txt
```

3. Configure environment variables
Set the `WANDB_ENTITY` environment variable to your Weights & Biases entity name.
```text
WANDB_ENTITY=`replace_with_your_wandb_entity`
```

3. Prepare datasets
Download datasets (REALDISP, REALWORLD, MM-Fit) as described in [data/README.md](data/README.md).

## 🧪 Running Scripts
### Individual Augmentation Assessment (Paper Section IV.B)
Run the following command to assess individual augmentations (e.g. MM-Fit dataset): 
```bash
# Movement amplitude scaling
python scripts/mmfit_ppda_indiv.py --magscale --magscale_sigma 0.2
# Movement amplitude warping
python scripts/mmfit_ppda_indiv.py --magwarp --magwarp_sigma 0.2 --magwarp_knot 4
# Movement speed scaling
python scripts/mmfit_ppda_indiv.py --timescale --timescale_scale_min 0.8 --timescale_scale_max 1.2
# Movement speed warping
python scripts/mmfit_ppda_indiv.py --timewarp --timewarp_sigma 0.1 --timewarp_knot 4
# Sensor rotation/placement variation
python scripts/mmfit_ppda_indiv.py --rotation --rotation_range_x -25,25 --rotation_range_y -25,25 --rotation_range_z -25,25
# Noise and bias addition
python scripts/mmfit_ppda_indiv.py --noisebias --noisebias_sigma 0.2
```

### Multi-Augmentation Assessment (Paper Section IV.C)
Run the following command to assess multiple augmentations (e.g. MM-Fit dataset):
```bash
# Run with default configuration (all augmentations enabled)
python scripts/mmfit_ppda_multi.py
```

Note: For other datasets, use the corresponding scripts (e.g., realdisp_ppda_indiv.py, realworld_ppda_multi.py).

## 📚 Citation
<!-- Include bibtex info -->

## 🙏 Acknowledgements

Parts of this repository are adapted from [dl_har_public](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public).  
We thank the authors for making their code available.

