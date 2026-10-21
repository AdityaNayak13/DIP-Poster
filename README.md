# Defective Product Detection System (DIP Pipeline Prototype)

An automated 8-stage visual inspection pipeline combining **Digital Image Processing (DIP)** and **Machine Learning** for automated quality assurance and defect detection.

---

## 🔬 8-Stage Inspection Pipeline

1. **Acquisition**: Image capture and standardized resolution normalization ($200 \times 200$).
2. **Enhancement**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) for illumination normalization.
3. **Restoration**: Gaussian spatial filtering to attenuate high-frequency sensor noise.
4. **Preprocessing**: Morphological opening and closing filters to suppress background texture.
5. **Segmentation**: Otsu's automatic thresholding to isolate the product region of interest (ROI).
6. **Feature Extraction**: Quantitative descriptors including Sobel gradient edge density, intensity standard deviation, fill ratio, and boundary circularity.
7. **Classification**: Rule-based deviation scoring against a calibrated reference profile, or a trained Random Forest classifier.
8. **Output**: Accept/Reject verdict, defect anomaly score (0–100), and confidence metrics.

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install opencv-python numpy flask scikit-learn joblib
```

### 2. Interactive Web UI Prototype

Open `inspection-prototype.html` directly in your web browser (Chrome, Edge, Firefox) for an interactive, visual demonstration of all 8 pipeline stages with live canvas renders.

### 3. Python Backend

- **Run Default Inspection:**
  ```bash
  python "inspection protptype.py"
  ```
- **Inspect Specific Image:**
  ```bash
  python "inspection protptype.py" inspect <path_to_image>
  ```
- **Inspect and Save Intermediate Stages:**
  ```bash
  python "inspection protptype.py" inspect <path_to_image> --save-stages output_stages
  ```
- **Run REST API Server:**
  ```bash
  python "inspection protptype.py" serve --port 5000
  ```

---

## 📁 Repository Structure

```text
├── inspection protptype.py    # Python implementation of the 8-stage DIP pipeline & REST API
├── inspection-prototype.html  # Interactive web UI demonstration with step-by-step canvas rendering
├── reference_profile.json     # Feature profile of non-defective product reference
├── output_stages/             # Intermediate DIP stage visual outputs (enhanced, edges, mask, etc.)
└── README.md
```
