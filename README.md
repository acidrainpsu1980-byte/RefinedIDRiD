# Retinal Lesion Segmentation with DeepLabV3+ (ResNet-50)

This repository contains MATLAB code for semantic segmentation of diabetic retinopathy lesions and retinal anatomical structures using a **DeepLabV3+** network with a **ResNet-50** backbone.

The model is trained on color fundus images with 12 classes, including background, retina, optic disc, vessels, and multiple lesion types.

---

## Features

- DeepLabV3+ with ResNet-50 backbone (`deeplabv3plusLayers`)
- Multi-class semantic segmentation with 12 classes
- Data augmentation (flip, translation, rotation)
- Training/validation split using `imageDatastore` and `pixelLabelDatastore`
- Visualization of predictions vs. ground truth with class-specific colormap

---

## Class Definitions

The following classes and label IDs are used throughout the project:

```matlab
classNames = ["Background" "Retina" "FV" "Vessel" "OD" "VH" "EX" "IRMA" "HE" "NV" "CWS" "MA"];
labelIDs   = [0 8 16 24 32 4 63 96 127 166 191 255];
