# AI-Based Network Attack Forecasting from Network Traffic Data

> **SIH 2026 — Problem Statement SIH26153**  
> **Theme:** Blockchain & Cybersecurity  
> **Organization:** National Technical Research Organisation (NTRO)

## Overview

Traditional Intrusion Detection Systems (IDS) primarily focus on identifying whether
network traffic is malicious at the current point in time.

This project explores a different question:

> **Given recent network behaviour, can we forecast the likelihood of malicious activity in the near future?**

The system learns temporal patterns from historical network traffic and produces
multi-horizon attack-risk forecasts for the next **30, 60, 90, 120 and 150 seconds**.

The current prototype uses network-state representations derived from **CIC-IDS2018**
traffic data and an **LSTM-based temporal model**.

The project is being developed as part of **Smart India Hackathon 2026** for
Problem Statement **SIH26153**.

---

## Problem Statement

### SIH26153 — AI based Network Attack Forecasting from Network Traffic Data

The objective is to learn evolving network-state dynamics from traffic telemetry and
forecast the likelihood and progression of malicious activity before compromise.

The intended system should:

- Learn temporal network-state transitions
- Forecast future network behaviour
- Predict attack likelihood over multiple future windows
- Generalize to attack patterns not seen during training
- Map predicted behaviour to recognized attack stages
- Provide explanations for predictions
- Support network traffic provided as CSV and/or PCAP

The current implementation focuses primarily on the **temporal forecasting component**
using flow-derived network-state representations.

---

# Proposed Approach

The current ML pipeline is:

```text
Network Traffic
      ↓
Data Cleaning & Validation
      ↓
Flow-Level Feature Processing
      ↓
30-Second Network-State Aggregation
      ↓
Temporal Window Construction
      ↓
Baseline Models
      ↓
LSTM Temporal Forecasting
      ↓
Multi-Horizon Attack-Risk Prediction
      ↓
Inference Output
      ↓
Dashboard Integration
```

The model uses the previous **10 × 30-second network states**
(approximately **5 minutes of historical context**) to forecast attack presence at:

- **+30 seconds**
- **+60 seconds**
- **+90 seconds**
- **+120 seconds**
- **+150 seconds**

---

# Dataset

The primary dataset used in the current prototype is:

**CIC-IDS2018**

CIC-IDS2018 contains network traffic generated under benign conditions and
multiple attack scenarios.

The processed traffic data contains flow-level features extracted using
**CICFlowMeter**.

## Dataset Processing

Across the 10 CIC-IDS2018 traffic files:

- Approximately **16.23 million network-flow records** processed
- Repeated CSV headers removed
- Invalid timestamps removed
- Records sorted chronologically
- 30-second network-state windows constructed
- Attack metadata mapped to corresponding windows
- Temporal gaps identified and preserved rather than artificially filled
- Chronological train/validation/test split used

---

# Network State Representation

Each 30-second window is represented using **24 state features**.

### Traffic Volume

- `Flow_Count`
- `Total_Fwd_Pkts`
- `Total_Bwd_Pkts`
- `Total_Fwd_Bytes`
- `Total_Bwd_Bytes`

### Flow Statistics

- `Mean_Flow_Duration`
- `Mean_Flow_Bytes_Rate`
- `Mean_Flow_Packets_Rate`
- `Mean_Flow_IAT`
- `Mean_Flow_IAT_Std`
- `Mean_Flow_IAT_Max`
- `Mean_Flow_IAT_Min`

### Packet Statistics

- `Mean_Fwd_Packets_Rate`
- `Mean_Bwd_Packets_Rate`
- `Mean_Packet_Length`
- `Mean_Packet_Length_Std`
- `Max_Packet_Length`
- `Min_Packet_Length`

### TCP / Protocol Behaviour

- `SYN_Count`
- `ACK_Count`
- `RST_Count`
- `PSH_Count`
- `Unique_Dst_Ports`
- `Unique_Protocols`

Attack labels and attack ratios are used as targets or metadata and are **not included
as model input features**.

> **Important:** The current 24-dimensional representation is primarily derived from
> flow-level/CICFlowMeter features. The SIH problem statement requires both
> flow-level and packet-level information. Packet-level feature integration is
> therefore part of the ongoing development.

---

# Forecasting Dataset

The forecasting model receives:

```text
10 historical windows × 24 features
```

representing approximately **5 minutes of historical network behaviour**.

The forecasting dataset contains:

- **9,198** valid sequence samples
- **10** historical states per sample
- **24** features per state
- **5** future prediction horizons
- No sequences crossing identified capture gaps

## Chronological Split

The data is split chronologically rather than randomly:

| Split | Samples |
|---|---:|
| Train | 5,957 |
| Validation | 1,110 |
| Test | 2,131 |

This prevents future traffic from being used to train models that are evaluated on
earlier traffic and reduces the risk of temporal data leakage.

---

# Models

## 1. Persistence Baseline

The previous attack state is used as the prediction for the next state.

This is an important baseline because attack activity in the dataset can persist for
multiple consecutive windows.

A strong persistence baseline is particularly important for this problem because
a model can achieve a high score simply by learning that an ongoing attack will
probably continue.

Therefore, persistence performance is used as a reference when evaluating the
machine-learning models.

---

## 2. Logistic Regression

A non-temporal baseline using the current network-state representation.

The preprocessing pipeline uses:

- Train-only median imputation
- Train-only standardization
- Class weighting
- Logistic regression

This baseline tests whether the current network state alone is sufficient for
future attack prediction.

---

## 3. LSTM

The main temporal prototype uses a Long Short-Term Memory (LSTM) network.

```text
Input: 10 × 24
        ↓
LSTM (64 units)
        ↓
Dropout (0.2)
        ↓
Dense (32, ReLU)
        ↓
Dense (5, Sigmoid)
        ↓
Attack probabilities at:
+30 / +60 / +90 / +120 / +150 seconds
```

The model is trained using:

- Chronological train/validation/test splits
- Train-only preprocessing
- Class-weighted binary cross-entropy
- Adam optimizer
- Early stopping
- Learning-rate reduction
- Gradient clipping

---

# Evaluation

Because this problem is strongly affected by temporal persistence, the evaluation
does not rely on a single accuracy value.

The project evaluates:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- False Positive Rate (FPR)

The persistence baseline is also reported to provide context for the machine-learning
results.

## LSTM Test Results

The current LSTM prototype produced the following results on the chronological
test set:

| Horizon | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| +30s | 0.894 | 0.768 | 0.827 | 0.878 | 0.895 |
| +60s | 0.926 | 0.255 | 0.400 | 0.797 | 0.822 |
| +90s | 0.955 | 0.452 | 0.614 | 0.882 | 0.891 |
| +120s | 0.823 | 0.232 | 0.362 | 0.668 | 0.650 |
| +150s | 0.904 | 0.613 | 0.731 | 0.838 | 0.870 |

These results demonstrate that the temporal model learns useful signal from the
network-state history.

However, the current LSTM **does not outperform the persistence baseline** on the
primary binary future-attack task.

This is an important finding rather than a result that is being hidden. It indicates
that the current target is strongly dominated by attack persistence and that a more
meaningful forecasting formulation is required.

---

# Early-Warning Evaluation

A separate evaluation was performed to test whether the model could predict an
attack **before its exact onset**, rather than simply predicting that an already
active attack would continue.

The early-warning task considers only windows where the current state is benign and
asks whether a new attack onset occurs within future horizons.

The current early-warning experiments used:

- Logistic Regression
- LSTM
- Chronological train/validation/test splits
- Exact attack-onset labels
- Multiple future horizons

The current models did **not generalize reliably to the chronological test set**.

This result is important because it shows that predicting attack continuation is
substantially easier than predicting previously unseen attack onset.

Future work therefore focuses on improving the state representation, incorporating
packet-level information, and reformulating the forecasting problem around learned
state transitions and attack progression.

---

# Attack Behaviour Analysis

Attack metadata was mapped to the 30-second network-state windows to analyze
different attack scenarios.

The current chronological evaluation includes attack patterns that are not present
in the training period.

Examples include:

- Bot
- Infiltration
- Brute Force
- DDoS
- DoS
- Web attacks
- SSH/FTP brute force
- Slowloris
- SlowHTTPTest

The current attack-family grouping is used only for analysis and visualization.

> It should **not** be interpreted as an official MITRE ATT&CK classification.

Formal attack-stage mapping using recognized frameworks remains future work.

---

# Inference

The repository contains an inference pipeline that accepts a sequence of
**10 network states × 24 features** and produces multi-horizon attack probabilities.

Example:

```bash
python src/forecasting/predict.py \
  --input data/sample/sample_sequence.csv \
  --model_dir results/lstm_baseline \
  --output data/sample/predictions.csv
```

On Windows PowerShell:

```powershell
python src/forecasting/predict.py `
  --input data/sample/sample_sequence.csv `
  --model_dir results/lstm_baseline `
  --output data/sample/predictions.csv
```

The inference pipeline:

1. Loads the input sequence
2. Validates the feature dimensions
3. Loads the trained preprocessing artifacts
4. Scales the network-state sequence
5. Runs the LSTM model
6. Produces probabilities for five future horizons
7. Applies horizon-specific thresholds
8. Produces predicted attack/benign states
9. Generates a UI-oriented risk level

Example output:

```text
+30s   46.21%   ATTACK   MEDIUM
+60s   42.68%   BENIGN   MEDIUM
+90s   42.18%   BENIGN   MEDIUM
+120s  47.13%   BENIGN   MEDIUM
+150s  53.92%   ATTACK   MEDIUM
```

The risk level is intended as a **dashboard/UI representation** and should not be
interpreted as a calibrated cybersecurity severity score.

---

# Project Structure

```text
SIH26153/
│
├── data/
│   ├── sample/
│   │   ├── sample_network_state.csv
│   │   ├── sample_forecasting.csv
│   │   └── sample_sequence.csv
│   │
│   ├── raw/
│   ├── interim/
│   ├── sorted/
│   └── processed/
│
├── docs/
│
├── notebooks/
│
├── results/
│   └── lstm_baseline/
│
├── scripts/
│
├── src/
│   ├── forecasting/
│   │   ├── build_lstm_sequences.py
│   │   ├── build_early_warning_dataset.py
│   │   ├── train_lstm_baseline.py
│   │   ├── predict.py
│   │   └── ...
│   │
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets, raw network captures, generated datasets, and large model artifacts
are excluded from the Git repository.

---

# Current Limitations

The current implementation is a **research/prototype stage**, not a production IDS.

The major limitations are:

### 1. Packet-Level Features

The current network state is primarily derived from flow-level CICFlowMeter
features.

The SIH problem statement explicitly requires both flow-level and packet-level
features.

Planned packet-level features include:

- TTL statistics
- TCP window statistics
- Fragmentation indicators
- Payload-size distributions
- Retransmission-related statistics
- Packet-level scanning/signature indicators

---

### 2. True State-Transition World Model

The current LSTM predicts future attack presence.

It does not yet implement the complete learned world-model formulation:

```text
P(S(t+1) | S(t))
```

with explicit multi-step state-transition simulation.

A future version will investigate whether the model can learn the evolution of the
network state itself rather than directly predicting only binary attack presence.

---

### 3. Early Attack-Onset Forecasting

The current early-warning experiments did not generalize reliably to unseen
chronological test conditions.

The current binary future-attack target is also strongly influenced by persistence.

Therefore, future work will focus on:

- Attack onset prediction
- Attack progression
- State-transition forecasting
- Longer and more meaningful prediction horizons
- Evaluation against unseen attack scenarios

---

### 4. Explainability

The current LSTM does not yet provide a complete feature-attribution or attention
based explanation layer.

Future versions will investigate methods such as:

- SHAP
- Integrated Gradients
- Attention-based explanations
- Feature attribution over historical windows

The goal is to answer:

> **Why does the model believe malicious activity is likely to occur?**

---

### 5. Attack-Stage Mapping

Formal mapping to MITRE ATT&CK or another recognized attack-stage framework has not
yet been implemented.

The current attack-type/family labels are dataset-derived and are used primarily
for analysis.

---

### 6. PCAP Processing

The current pipeline primarily operates on processed CSV traffic data.

A future version will support:

```text
PCAP
 ↓
Packet Extraction
 ↓
Flow + Packet Features
 ↓
Network-State Construction
 ↓
Temporal Forecasting
```

---

# Future Work

The planned development roadmap is:

```text
Current Flow-Based State Representation
                ↓
Packet-Level Feature Extraction
                ↓
Combined Flow + Packet Network State
                ↓
Improved Temporal Representation
                ↓
Learned State-Transition Model
                ↓
Multi-Step Forward Simulation
                ↓
Attack Progression Forecasting
                ↓
MITRE ATT&CK Stage Mapping
                ↓
Explainable Predictions
                ↓
Real-Time / Streaming Inference
                ↓
Dashboard Integration
```

Potential model architectures include:

- LSTM / GRU
- Temporal Transformers
- Graph Neural Networks
- Latent-state world models
- Hybrid temporal + graph architectures

More complex architectures will only be introduced if they provide measurable
improvement over simpler baselines.

---

# Reproducibility

The project follows several measures to make experiments reproducible:

- Fixed random seeds where applicable
- Chronological dataset splits
- Train-only preprocessing
- Explicit feature ordering
- Saved preprocessing artifacts
- Saved model configuration
- Documented evaluation metrics
- Separate baseline and temporal-model experiments

The repository does not contain the full CIC-IDS2018 dataset because of its size.

The dataset should be obtained separately from its official source and placed in the
appropriate local data directory.

---

# Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Machine Learning

- Scikit-learn
- TensorFlow / Keras

### Network Analysis

- CICFlowMeter
- Planned: Scapy / PyShark for packet-level processing

### Development

- Git
- GitHub
- VS Code

### Planned Interface

- Streamlit / Flask
- Web-based dashboard

---

# Key Design Principles

The project follows several principles during development:

### No Random Temporal Splitting

Network traffic is time-dependent. Randomly mixing future traffic into training data
can produce misleadingly strong results.

### No Target Leakage

Attack labels and target metadata are not used as input features.

### Baselines Before Complex Models

A persistence baseline and classical machine-learning baseline are evaluated before
introducing more complex temporal architectures.

### Measure Before Claiming

A high metric is not automatically evidence of successful attack forecasting.

The project distinguishes between:

- Attack continuation
- Future attack presence
- Attack onset
- Early warning
- Attack progression

These are different prediction problems and are evaluated separately.

---

# Project Status

### Completed

- [x] CIC-IDS2018 dataset acquisition
- [x] Dataset cleaning and validation
- [x] Timestamp normalization
- [x] Chronological sorting
- [x] 30-second network-state construction
- [x] Attack-window mapping
- [x] Temporal forecasting dataset
- [x] Chronological train/validation/test split
- [x] Persistence baseline
- [x] Logistic regression baseline
- [x] LSTM forecasting prototype
- [x] Multi-horizon prediction
- [x] Attack episode analysis
- [x] Early-warning dataset
- [x] Early-warning baseline experiments
- [x] Inference pipeline
- [x] GitHub repository setup

### In Progress

- [ ] Packet-level feature extraction
- [ ] Combined flow + packet network state
- [ ] Learned state-transition/world model
- [ ] Multi-step state simulation
- [ ] Explainability
- [ ] Formal attack-stage mapping
- [ ] Dashboard integration
- [ ] PCAP-based inference
- [ ] Improved early-warning evaluation

---

# SIH Problem Statement

This project is developed for:

**Smart India Hackathon 2026**

**Problem Statement:** SIH26153  
**Title:** AI based Network Attack Forecasting from Network Traffic Data  
**Organization:** National Technical Research Organisation (NTRO)  
**Theme:** Blockchain & Cybersecurity  
**Category:** Software

---

# Disclaimer

This repository contains a research and hackathon prototype.

The current model should **not** be considered a production-ready intrusion
detection or cyber-defense system.

Model predictions depend on the training data, feature representation, preprocessing
pipeline, and evaluation setup.

Further validation on diverse real-world network environments is required before
any operational cybersecurity deployment.

---

# License

This project is developed as part of an academic/hackathon project for
**Smart India Hackathon 2026**.

Refer to the repository for project-specific licensing information.
