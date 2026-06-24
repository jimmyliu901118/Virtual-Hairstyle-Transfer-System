# Virtual-Hairstyle-Transfer-System
A hybrid virtual try-on pipeline combining WGAN-GP and dlib geometric alignment.

This is the official repository for our senior graduation project at National Yang Ming Chiao Tung University (NYCU). 

## 📌 Project Overview
In this project, we developed a hybrid virtual try-on pipeline that decouples hairstyle generation from facial identity preservation. This engineering trade-off allowed us to achieve robust and high-resolution blending results under limited cloud-GPU computing resources.



## 🛠️ System Architecture & Codebase
The project is structurally divided into three core stages:

1. **Facial Landmark Extraction (`point_detection.py`)**
   - Utilizes `dlib`'s HOG-based face detector and 68-landmark shape predictor.
   - Automatically extracts facial coordinates (eyes, nose, forehead contours) to serve as geometric bridges.
2. **Generative Modeling (`GAN_hairstyle.ipynb`)**
   - Implements a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** framework.
   - Built with PyTorch to mitigate mode collapse and vanishing gradients, training on 70k+ images to synthesize high-quality hairstyle assets from scratch.
3. **Geometric Alignment & Compositing (`pasting_blending.py`)**
   - Leverages OpenCV to calculate **affine transformation matrices** based on landmark anchors (eyes and nose center).
   - Performs spatial warping to align generated hair onto real target faces, followed by **Gaussian alpha blending** for smooth edge transitions.

## 👥 Contributors
- 劉昱君 (NYCU CS)
- 高永杰 (NYCU CS)
- 黃文彥 (NYCU CS)
