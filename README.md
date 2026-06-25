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
2. **Generative Modeling (`GAN_Virtual_Hairstyle.ipynb`)**
   - Implements a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** framework.
   - Built with PyTorch to mitigate mode collapse and vanishing gradients, training on 70k+ images to synthesize high-quality hairstyle assets from scratch.
3. **Geometric Alignment & Compositing (`pasting_blending.py`)**
   Leverages OpenCV to calculate affine transformation matrices based on facial anchors (eyes and nose center). Performs spatial warping followed by Gaussian blending. **Features a custom "Arch-Cutting Mask" and "Translation Compensation" algorithm to discard the GAN image's foreign forehead skin while ensuring a perfect, artifact-free fit onto the target face.**


## 🚀 Quick Start (Execution Pipeline)

To replicate our virtual try-on results, you must follow the pipeline sequentially. **Crucially, the landmark extraction script must be executed twice**—once for the target face and once for the generated hairstyle image—to establish the coordinate mapping for composition.

### Step 1: Extract Landmarks for Target Face (User)
Run feature extraction on the real face image (e.g., `Messi.jpg`). This generates a coordinate text file used as the geometric destination anchor.

```bash
python point_detection.py --input Messi.jpg --output Messi.txt

```markdown
### Step 2: Extract Landmarks for Generated Hairstyle (GAN)
Run the script a second time on the synthesized hair image (e.g., `hair_50.jpg`). This generates the source anchor coordinates.

```bash
python point_detection.py --input hair_50.jpg --output hair_50.txt

```markdown
### Step 3: Run Geometric Alignment & Translation Compositing
Execute the final blending pipeline. The script will apply an affine transformation based on both coordinate files, invoke **Arch-Cutting** to discard the GAN image's forehead skin, and apply **Translation Compensation ($M_{translate}$)** to lower the hair into a seamless fit.

```bash
python pasting_blending.py --real_face Messi.jpg --real_txt Messi.txt --gan_hair hair_50.jpg --gan_txt hair_50.txt

## 👥 Contributors
- 劉昱君 (NYCU CS)
- 高永杰 (NYCU CS)
- 黃文彥 (NYCU CS)


