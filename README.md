# CLaC at SemEval-2025 Task 6: Corporate Environmental Promise Verification

This repository contains the implementation, model artifacts, and poster for our system submitted to **PromiseEval @ SemEval-2025**.

<p align="center">
  <img src="SemEval_2025_Poster.png" alt="CLaC at SemEval-2025 Poster" width="80%">
</p>

---

## Paper  
**Nawar Turk, Eeham Khan, & Leila Kosseim (2025)**  
*CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification.*  
In *Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)*, Vienna, Austria.  
[Published Paper (ACL Anthology)](https://aclanthology.org/2025.semeval-1.232)

---

## Repository Structure
- **Model_1/** → Base Model implementation (ESG-BERT with task-specific classifier heads)
  - **model-artifacts/** → JSON configs & hyperparameters for each of the 4 subtasks
- **Model_3/** → Combined Subtask Model (DeBERTa-v3-large with attention pooling and metadata enrichment)

---

## Model Descriptions
- **Model 1 - Base Model (ESG-BERT)**  
  Uses ESG-BERT with four subtask-specific classifier heads (Promise Identification, Evidence Detection, Clarity, and Timing). Fine-tunes the last two transformer layers and classification heads while freezing earlier layers to reduce computational cost and overfitting risk.

- **Model 3 - Combined Subtask Model (DeBERTa-v3-large)**  
  Implements multi-task learning for Promise and Evidence subtasks, with attention pooling, metadata enrichment, and dual classifier heads. This model achieved our highest private leaderboard score.

---

## Links
- [Google Colab Notebook with Results & Experiment Logs](https://colab.research.google.com/drive/1qlOs2B7PWvADnD3TaIluonRC5XqfmJbG?usp=sharing)
- [Model 1 Jupyter Notebook with Results & Experiment Logs](https://huggingface.co/datasets/nawarturk/SemEval2025-Task6/tree/main)
- [Task Overview](https://sites.google.com/view/promiseeval/promiseeval)
- [Poster (High-Resolution)](SemEval_2025_Poster.png)

