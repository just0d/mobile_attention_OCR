# Handwritten Name Recognition - Lightweight CRNN for Mobile Devices

This repository contains the code and final report for Machine Learning I course project: a compact CRNN-based model for recognizing handwritten names. The model is optimized for mobile deployment by combining MobileNetV3 with a BiLSTM decoder and multi-head attention.

---

## Key Features

- Optimized for mobile devices  
- 10.9M parameters (10x smaller than EasyOCR)  
- Trained on IAM dataset (handwritten English names)  
- Real-time performance on smartphones  
- Includes preprocessing, training, and evaluation scripts  

---

## Model Architecture

- **Encoder**: MobileNetV3 (input: `64x256x1`)  
- **Decoder**: BiLSTM + Multi-head Attention  
- **Output**: Character sequences with [Start], [End], and [PAD] tokens  

---

## Dataset

- **Source**: IAM Handwriting Dataset  
- **Size**: ~330,000 character images  
- **Writers**: 657 individuals, 13,353 text lines  
- **Preprocessing**:  
  - Resized to `64x256`  
  - Centered on canvas  
  - OTSU binarized and morphologically cleaned  

---

## Training

- Cosine learning rate decay with warmup  
- Adam optimizer with gradient clipping  
- Regularization (L1/L2) on LSTM weights  
- Early stopping & LR scheduling via Keras callbacks  
- Trained on Google Colab with GPU (T4)  

---

## Evaluation

- Sequence Accuracy (case-insensitive)  
- Edit Distance (mean and std)  
- MAE & MSE of predicted string lengths  
- Bias-variance gap per epoch  
- Training/validation accuracy and loss logging  

---

## Project Structure

src/
├── model.py           # MobileNetV3 + BiLSTM + Attention model definition  
├── train.py           # Training loop, learning rate scheduling, and training logic  
├── preprocess.py      # IAM dataset parsing, filtering, and preprocessing  
├── data_generator.py  # Custom tf.keras Sequence generator  
├── utils.py           # Training tracker, evaluation metrics, and visualizations  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/just0d/mobile_attention_OCR.git
   cd mobile_attention_OCR

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare IAM dataset under:
   /content/drive/MyDrive/iam_data/

## Training

To train the model: 
python src/train.py
Outputs:
- Model checkpoints
- Training logs (final_training_history.csv)
- Evaluation metrics

## License

This project is licensed under the MIT License. See LICENSE for details

## Acknowledgements
- IAM Handwriting Dataset
- TensorFlow/Keras
- MobileNetV3 authors
- EasyOCR & academic CRNN baselines

## Authors

Created by students of Hanyang University, Department of Data Science.

## References

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv. Retrieved from https://arxiv.org/abs/1409.0473

Howard, A., Sandler, M., Chu, G., et al. (2019). Searching for MobileNetV3. arXiv. Retrieved from https://arxiv.org/abs/1905.02244

Shi, B., Bai, X., & Yao, C. (2017). An end-to-end trainable neural OCR approach. SCITEPRESS – Science and Technology Publications. Retrieved from https://doi.org/10.5220/0006123703210328

Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, Tengyu Ma.(2024).Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective. Retrieved from https://arxiv.org/abs/2410.05192

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. Retrieved from https://arxiv.org/pdf/1905.02244

Marti, U. V., & Bunke, H. (2002). "The IAM-database: an English sentence database for offline handwriting recognition." International Journal on Document Analysis and Recognition. Retrieved from https://link.springer.com/article/10.1007/s100320200071

---
