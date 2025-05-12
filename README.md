# short-term-colonization-periphyton
# Supplementary Materials – Short-term Colonization by Pseudo-periphyton under Thermal Stress

This repository contains the supplementary data and analysis scripts for the study *“Short-term colonization dynamics of pseudo-periphyton under thermal stress”*.

## Contents

- `data/` – Raw and processed data used in the analyses
- `scripts/` – R/Python scripts used for data processing and analysis

## How to Use

1. Clone or download this repository.
2. Open and run the scripts using R or Python (see software requirements below).
3. Ensure the data files are kept in their original folder structure.

## Software Requirements

## Requirements

- Python ≥ 3.8
- Tested in Google Colab and local environments
- Recommended RAM: at least 8 GB if processing high-resolution TIFFs
- Images are processed in blocks due to memory usage (adjust block size manually in script)

## Input Image Requirements

To ensure consistent and reproducible analysis, input images should meet the following conditions:

- **Format**: `.tif` or `.tiff`. Original photographs should be taken in RAW format and later converted to TIFF without compression artifacts.
- **Lighting**: All samples must be photographed under uniform lighting conditions to avoid shadows, glare, or exposure variation.
- **Camera setup**: Use the same height and angle for all images. Avoid operator interference (e.g., shadows or reflections).
- **Background**: Place samples on a **flat and level surface** to prevent water or biofilm accumulation that could bias image interpretation.
- **Framing**: Ensure that the area of interest is fully captured and centered within the frame.
- **File naming**: Use unique and consistently formatted filenames (e.g., `day01_sampleA.tiff`) for automated processing.
**Sample size**: Images must capture the full 5×5 cm surface of the experimental plate. Smaller areas or cropped subregions (e.g., half plates) are not recommended, as they result in inconsistent and non-comparable metric outputs.

## Citation

If you use this material, please cite the original article:  
**[Citation will be added upon publication]**

## License

This material is provided for academic and research purposes only.
