# Semi-Supervised Anomaly Detection for Network Intrusion Detection

This project implements a semi-supervised learning pipeline for detecting network intrusions using both autoencoders and a classical anomaly detection technique.  The system is trained and evaluated on the CIC-DDoS2019 dataset provided by the Canadian Institute for Cybersecurity.

The aim is to identify malicious traffic using models trained just on benign data, which reflects a common scenario in real-world cybersecurity where labeled attack data is often limited or unavailable. Additionally, since new types of attacks frequently emerge, the system must be capable of detecting previously unseen threats.

---

## Dataset

We use the CIC-DDoS2019 dataset provided by the Canadian Institute for Cybersecurity.

- Source: [CIC DDoS 2019](https://www.unb.ca/cic/datasets/ddos-2019.html)

The dataset is organized into two main folders, each corresponding to network traffic collected on a different day:

* **`CSV_01_12`**: Used **for training**. It contains raw network traffic with both benign and attack flows. Only BENIGN flows are extracted from this folder and aggregated into a file named `benign_01_12.csv`, which is used to train the models.

* **`CSV_03_11`**: Used **for testing**. It also contains raw network traffic with both benign and attack flows. From this folder, new test datasets are created, each containing benign traffic combined with a single attack type, while preserving the original timestamps of the traffic flows. These processed datasets are saved in the `CICDDoD2019_test_new/` directory and are generated using the `data_preprocessing.py` script.

---

## 1. Data Preprocessing

**File:** `data_preprocessing.py`

This script is designed to better understand the characteristics of the data and prepare it for training the anomaly detection models. It performs the following tasks:

- **Flow Label Counting**  
  Counts the number of flows per class label across multiple CSV files to assess the distribution of benign versus attack traffic.

- **Benign Traffic Extraction**  
  Extracts and aggregates only benign flows from `CSV_01_12` for training. The output is saved as `benign_01_12.csv`.

- **Feature Selection**  
  Selects top-k features per attack type by combining:
  - Variance Thresholding
  - Random Forest feature importance
  - Mutual Information
  - F-score  
  These scores are averaged into a unified feature ranking.

- **Attack Dataset Segmentation**  
Extracts specific attack types from `CSV_03_11` to create clean and focused test datasets. Each resulting file contains benign traffic alongwith flows of a single attack type, while preserving the original timestamps. These are saved in the `CICDDoD2019_test_new/` directory.

- **Timestamp Analysis**  
  Reports first and last timestamps in each dataset to ensure temporal consistency.

- **t-SNE Visualization**  
  Applies dimensionality reduction for visualizing benign versus attack traffic separation.

---

## 2. Semi-Supervised Learning

**File:** `semi_supervised.py`

Implements the following models for anomaly detection:

- **One-Class SVM**  
  A classical semi-supervised anomaly detection algorithm.

- **Autoencoders**  
  - Autoencoder 0: A shallow model with a single latent layer.  
  - Autoencoder 1: A deeper, symmetrical architecture for improved reconstruction.  
  - Autoencoder 2: Sequence-based model using LSTM layers to capture temporal dependencies.

Uses reconstruction loss (for autoencoders) or decision function (for One-Class SVM) to detect anomalies. Performance is evaluated using accuracy, F1 score, precision, and recall.

---

## Notes

- Adjust dataset file paths according to your directory structure.  
- Due to the size of the dataset, data is loaded and processed in chunks to avoid memory issues.
