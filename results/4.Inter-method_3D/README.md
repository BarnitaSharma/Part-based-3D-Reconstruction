# Inter-Method 3D Analysis

This directory contains data and results used for **inter-method comparison**
between the proposed part-based 3D reconstruction pipeline and
Structure-from-Motion and Multiview Stereo (SfM-MVS) based reconstructions.

The analysis evaluates **internal consistency across reconstruction methods**
and **geometric accuracy** with respect to a synthetic CAD reference model.

---

## Data Sources

This directory uses the following precomputed data for inter-method evaluation:

- **Segmented sparse and dense point clouds**<sup>[1,2,3,4]</sup>  
  *(semantic labels obtained using SfM-guided SAM)*

- **Synthetic CAD reference model**<sup>[5]</sup>

- **Semantic voxel grid produced by the proposed pipeline**





All SfM-based point clouds are generated from the PhotoTourism Taj Mahal dataset
and are provided in segmented form within this directory.

---

## Processing Overview

The inter-method analysis follows the steps below:

1. Load segmented sparse and dense SfM point clouds  
2. Crop the dense reconstruction to the sparse bounding box  
3. Estimate the dominant facade plane from the segmented sparse cloud and align it to the global Z-axis  
4. Generate a naive four-way symmetric completion  
5. Refine symmetry using ordered ICP alignment  
6. Load the semantic voxel grid produced by the proposed pipeline  
7. Load and align a synthetic CAD reference model  
8. Align all reconstructions to a common vertical (Y-axis) reference  

All processing steps are implemented in:
utils/preprocess_helpers.py

---

## References and External Resources

[1] **PhotoTourism – Taj Mahal dataset**  
Jin, Y., Mishkin, D., Mishchuk, A., et al.  
*Image Matching Across Wide Baselines: From Paper to Practice*  
International Journal of Computer Vision, 129:517–554  
https://www.cs.ubc.ca/~kmyi/imw2020/data.html

[2] **COLMAP (Structure-from-Motion and Multi-View Stereo)**  
Schönberger, J. L., and Frahm, J.-M.  
*Structure-from-Motion Revisited*, CVPR 2016  
Schönberger, J. L., Zheng, E., Pollefeys, M., and Frahm, J.-M.  
*Pixelwise View Selection for Unstructured Multi-View Stereo*, ECCV 2016  
https://github.com/colmap/colmap

[3] **SfM-guided SAM**  
Sharma, B., and Jana, S.  
*Semantic Part Segmentation of Heritage Monuments from Crowdsourced Images:  
Consistent Labeling using SfM-guided SAM*  
National Conference on Communications (NCC), 2025

[4] **Synthetic CAD reference model**  
*Free3D Contributors*. *Taj Mahal 3D Model*.  
https://free3d.com/3d-model/taj-mahal-82618.html  

---

## Notes

- This analysis is used exclusively for evaluation and comparison
- It does not affect the core reconstruction pipeline
- No learning-based methods are used at this stage

