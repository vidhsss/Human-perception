# Human Perception Research Project

A comprehensive research project investigating human visual perception phenomena using computer vision models. This project compares how different neural network architectures, particularly self-supervised models like Masked Autoencoders (MAE), respond to various visual illusions and perceptual effects that are well-documented in human psychology.

## Overview

This repository contains experiments that test whether self-supervised networks exhibit similar perceptual biases and effects as humans when processing visual stimuli. This work extends the foundational research by [Jacob et al. (2021)](https://www.nature.com/articles/s41467-021-22078-3) by focusing specifically on **self-supervised learning models** and their perceptual capabilities.

The research covers multiple perceptual phenomena including the Thatcher effect, mirror confusion, relative size effects, 3D perception, occlusion, object parts, and global-local processing, building upon the comprehensive framework established in the original study.

## Project Structure

### Experiments

The project is organized into 12 main experiments, each focusing on a specific perceptual phenomenon:

#### **exp01_thatcher_effect/**
- **Phenomenon**: Thatcher Effect - the difficulty in recognizing facial features when faces are inverted
- **Files**: 
  - `CheckThatcherEffect.py` - Main analysis function
  - `mainCodeThatcherEffect_allNetworks.py` - Cross-network comparison
  - `tatcherFaces.mat` - Dataset with 80 face images (20 people, 4 conditions each)
- **Analysis**: Calculates Thatcher index using Euclidean distance across network layers

#### **exp02_mirror-confusion/**
- **Phenomenon**: Mirror Confusion - how networks handle mirrored versions of images
- **Files**:
  - `mirror.py` - Python implementation
  - `CheckMirrorConfusion.m` - MATLAB analysis
  - `mainCodeMirrorConfusion_allNetworks.m` - Multi-network comparison
- **Analysis**: Compares responses to original vs. horizontally/vertically mirrored images

#### **exp07_rel-size/**
- **Phenomenon**: Relative Size Effects - how size relationships affect perception
- **Files**:
  - `rel_size.py` - Python implementation with VGG-16
  - `relSize.mat` - Stimulus dataset
- **Analysis**: Tests modulation index across 12 groups of tetrads

#### **exp08_rsurf/**
- **Phenomenon**: Surface-based processing effects
- **Files**:
  - `mainCodeRsurf_allNetworks.m` - Multi-network analysis
  - `rsurf.mat` - Stimulus data

#### **exp09_3D/**
- **Phenomenon**: 3D Shape Perception
- **Files**:
  - `3d.py` - Python implementation
  - `3d.mat` - 3D stimulus dataset
- **Analysis**: Compares Y-shapes vs. cuboids, squares vs. cuboids

#### **exp10_occlusion/**
- **Phenomenon**: Occlusion Effects - how partial visibility affects recognition
- **Files**:
  - `occ.py` - Python implementation
  - `occlusion_set1.mat` - Occlusion stimulus dataset
- **Analysis**: Tests basic occlusion effects and depth ordering

#### **exp11_object_parts/**
- **Phenomenon**: Object Parts and Part Summation
- **Files**:
  - `main_code_PartSummation_AllNetworks.m` - Part summation analysis
  - `mainCode_PartXuSingh_AllNetworks.m` - Xu & Singh model comparison
  - `natunat_stim.mat`, `xustim.mat` - Stimulus datasets

#### **exp12_global-local/**
- **Phenomenon**: Global vs. Local Processing
- **Files**:
  - `global.py` - Python implementation
  - `GL.mat` - Global-local stimulus dataset

### Library Functions (`lib/`)

Core utilities for feature extraction and analysis:

- **`extract_features.py`** - Extracts activations from neural network layers
- **`distance_calculation.py`** - Implements various distance metrics (Euclidean, CityBlock, Cosine)
- **`layerwise_mi_figures.py`** - Visualization utilities for layer-wise analysis
- **`plot_all_network.m`** - MATLAB plotting functions

### Masked Autoencoder (MAE) Implementation (`mae/`)

A complete PyTorch implementation of Masked Autoencoders for vision learning:
- Pre-training and fine-tuning scripts
- ViT-Base, ViT-Large, and ViT-Huge model support
- Visualization demos and utilities

## Key Features

### Multi-Network Analysis
- Tests across **9 different self-supervised learning models** spanning multiple architectures
- **Contrastive SSL Models**: DINOv2, iBOT, SimCLR, SWAV, DeepCluster, IPCL
- **Representational SSL Models**: MAE, BYOL, UNICOM, Auto-encoders
- **Architecture Coverage**: Vision Transformers (ViT), ResNet-50, VGG-16, AlexNet
- Compares self-supervised models with supervised baselines
- Focuses on how different SSL approaches affect human-like perception

### Distance Metrics
- Euclidean distance
- City Block (Manhattan) distance
- Cosine similarity
- Spearman and Pearson correlations

### Layer-wise Analysis
- Extracts features from all network layers
- Analyzes how perceptual effects change across network depth
- Compares early vs. late layer representations

### Comprehensive Experimental Framework
- **Shape-Texture Bias Analysis**: 6 different image manipulations (original, greyscale, sketch, silhouette, edges, stylized)
- **13 Perceptual Phenomena**: Thatcher effect, mirror confusion, scene incongruence, multiple objects, correlational sparseness, Weber's law, relative size, surface invariance, 3D processing, occlusion, depth, object parts, global advantage
- **Cue-Conflict Experiments**: Direct comparison of shape vs. texture bias
- **Cross-Architecture Comparison**: ViT, ResNet, VGG models with different SSL approaches

## Usage

### Running Experiments

1. **Setup Environment**:
   ```bash
   # Install required packages
   pip install torch torchvision scipy numpy matplotlib pillow opencv-python
   ```

2. **Run Individual Experiments**:
   ```python
   # Example: Thatcher Effect
   cd exp01_thatcher_effect/codes/
   python CheckThatcherEffect.py
   
   # Example: Mirror Confusion
   cd exp02_mirror-confusion/
   python mirror.py
   ```

3. **Feature Extraction**:
   ```python
   from lib.extract_features import extract_features
   from lib.distance_calculation import distance_calculation
   
   # Extract features from network
   features = extract_features(stimuli, network)
   
   # Calculate distances
   distance = distance_calculation(feature1, feature2, 'Euclidean')
   ```

### MATLAB Compatibility

Many experiments include both Python and MATLAB implementations for cross-platform compatibility:

```matlab
% Example: Running MATLAB version
cd exp01_thatcher_effect/codes/
mainCodeThatcherEffect_allNetworks
```

## Research Applications

This project is valuable for:

- **Computational Neuroscience**: Understanding how artificial networks compare to human vision
- **Computer Vision**: Improving model robustness and interpretability through self-supervised learning
- **Psychology**: Validating perceptual theories with computational models
- **AI Safety**: Understanding potential biases in vision systems
- **Self-Supervised Learning**: Advancing our understanding of how unsupervised learning affects perceptual representations

## Relationship to Foundational Work

This research builds upon the seminal work by [Jacob et al. (2021)](https://www.nature.com/articles/s41467-021-22078-3) which systematically compared visual object representations between brains and deep networks. The original study found that:

- **Present in trained networks**: Thatcher effect, mirror confusion, Weber's law, relative size, multiple object normalization
- **Absent in trained networks**: 3D shape processing, surface invariance, occlusion, natural parts, global advantage
- **Present in random networks**: Global advantage effect, sparseness, relative size

**Our extension** investigates whether these findings hold for self-supervised models, particularly:
- How MAE and other self-supervised models compare to supervised models on these perceptual tasks
- Whether self-supervised learning can capture perceptual phenomena that supervised learning misses
- The role of training paradigm in the emergence of human-like perception

## Key Findings

The experiments reveal how different neural network architectures respond to:
- **Face inversion effects** (Thatcher effect)
- **Mirror symmetry** processing
- **Size constancy** and relative size effects
- **3D shape** perception
- **Occlusion** and depth ordering
- **Part-whole** relationships
- **Global vs. local** processing

## Final Study Results

This work extends the comprehensive analysis by [Jacob et al. (2021)](https://www.nature.com/articles/s41467-021-22078-3) by investigating how **self-supervised learning models** compare to supervised models in exhibiting human-like perceptual phenomena.

### Key Research Questions

Building on the original study's findings, this research addresses:
- Do self-supervised models (like MAE, DINOv2, SimCLR) exhibit the same perceptual phenomena as supervised models?
- How does the training paradigm (self-supervised vs. supervised) affect the emergence of human-like perception?
- Which perceptual effects are architecture-dependent vs. training-dependent in self-supervised models?
- How do contrastive learning and representational learning approaches differ in capturing human perception?

### Major Findings

#### Shape-Texture Bias Analysis
- **Human observers**: Exhibit pronounced shape bias for object categorization
- **ResNet-based SSL networks**: Show strong texture bias (similar to supervised ImageNet models)
- **Vision Transformer (ViT) models**: Display equal bias towards both texture and shape, bridging the gap between human perception and computational models

#### Perceptual Phenomena in Self-Supervised Models

**Contrastive SSL Models** (DINOv2, iBOT, SimCLR, SWAV, DeepCluster, IPCL):
- ✅ **Strong capabilities**: Mirror confusion, scene incongruence, correlational sparseness, surface invariance, Weber's law, 3D processing
- ❌ **Limitations**: Occlusion perception, object parts recognition

**Representational SSL Models** (MAE, BYOL, UNICOM, Auto-encoders):
- ✅ **Strengths**: Mirror confusion, scene incongruence, correlational sparseness, global advantage
- ❌ **Weaknesses**: 3D processing, Weber's law, relative size perception, surface invariance

#### Architecture-Specific Findings

**ViT-based models**: Lower texture bias, better shape-texture balance
**ResNet-based models**: Strong texture bias, better 3D processing capabilities
**VGG-based models**: Mixed performance, some show surface invariance when trained on stylized datasets

### Research Implications

This work provides crucial insights into:
- **Self-supervised learning** can capture broader range of human perception phenomena than supervised learning
- **Contrastive SSL** generally outperforms representational SSL in capturing perceptual effects
- **Architecture choice** significantly influences perceptual biases and capabilities
- **Training on stylized datasets** can reduce texture bias and improve shape-based perception
- **Current limitations**: All models struggle with occlusion and object parts recognition

## Dependencies

### Python
- PyTorch
- Torchvision
- NumPy
- SciPy
- Matplotlib
- PIL (Pillow)
- OpenCV

### MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

## Citation

If you use this code in your research, please cite the relevant papers and acknowledge this repository.

### Foundational Work
This research extends the comprehensive study by Jacob et al.:

```bibtex
@article{jacob2021qualitative,
  title={Qualitative similarities and differences in visual object representations between brains and deep networks},
  author={Jacob, Georgin and Pramod, RT and Katti, Harish and Arun, SP},
  journal={Nature Communications},
  volume={12},
  number={1},
  pages={1872},
  year={2021},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-021-22078-3}
}
```

### Final Study
**"Exploring Parallels and Discrepancies Between the Biological Visual System and Self-Supervised Learning"**

Authors: Vidhi Jain, Pramod Kaushik, and Bapi Raju  
Institutions: Netaji Subhas University of Technology (NSUT), Delhi, India & International Institute of Information Technology (IIIT), Hyderabad, India

You can find the final paper at https://drive.google.com/file/d/1sfUV5zHaDu-a_y5zykZHHG5E78Uj7GMj/view?usp=sharing

```bibtex
@article{jain2024exploring,
  title={Exploring Parallels and Discrepancies Between the Biological Visual System and Self-Supervised Learning},
  author={Jain, Vidhi and Kaushik, Pramod and Raju, Bapi},
  journal={[Journal/Conference]},
  year={2024},
  note={Comprehensive analysis of self-supervised learning models and human perception phenomena}
}
```

## License

This project includes code from multiple sources:
- MAE implementation: CC-BY-NC 4.0 (see `mae/LICENSE`)
- Original research code: Please check individual experiment folders for specific licensing

## Contributing

This is a research repository. For questions or contributions, please contact the research team.

---

*This repository contains experimental code for investigating human perception phenomena using deep neural networks. The work bridges computational neuroscience, computer vision, and cognitive psychology.*