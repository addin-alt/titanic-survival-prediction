<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,35:0F766E,70:0284C7,100:38BDF8&height=240&section=header&text=Titanic%20Survival%20Predictor&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Machine%20Learning%20Pipeline%20%7C%20Feature%20Engineering%20%7C%20Gradient%20Boosting&descSize=16&descAlignY=60" />

<img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=800&size=24&duration=2800&pause=900&color=38BDF8&center=true&vCenter=true&width=950&lines=End-to-End+Classification+Pipeline;Gradient+Boosting+%2B+Advanced+Engineering;Scikit-Learn+%7C+Pandas+%7C+Python;84%25+Validation+Accuracy" />

<br/>

<a href="#">
  <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" />
</a>

<img src="https://img.shields.io/badge/Status-Completed-10B981?style=for-the-badge" />
<img src="https://img.shields.io/badge/Accuracy-84%25-7C3AED?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />

</div>


## ✨ Project Overview

**Titanic Survival Predictor** is a complete, end-to-end Machine Learning classification pipeline designed to predict passenger survival outcomes. 

By utilizing advanced feature engineering (such as title extraction and familial grouping) and robust data imputation strategies, the system effectively handles messy, real-world historical data. The processed dataset is evaluated using an optimized Gradient Boosting Classifier, achieving high precision and recall without overfitting.


## 🎯 Primary Objectives

* Extract hidden socioeconomic signals using regex-based `Title` extraction from unstructured text.
* Eliminate data gaps via median imputation grouping (e.g., estimating age based on passenger title).
* Prevent model hallucination/overfitting using rigorous Train/Validation holdout splits.
* Deliver high-accuracy binary classification using a Gradient Boosting architecture.
* Generate production-ready CSV outputs formatted strictly for the Kaggle evaluation leaderboard.


## 🛠 Data Pipeline & Feature Engineering

Real-world datasets are inherently messy. This project handles missing values and categorical data through a structured pipeline:

1. **Regex Title Extraction:** Parsed passenger names to isolate honorific titles (`Mr`, `Mrs`, `Miss`, `Master`, and `Rare`). Titles strongly correlate with socioeconomic class and historical evacuation priorities.
2. **Smart Median Imputation:** Instead of a generic global median, missing ages were imputed dynamically using the median age *within each specific title group*.
3. **Family Dynamics Features:** Combined `SibSp` and `Parch` variables to compute total `FamilySize` and isolated solo travelers via an `IsAlone` boolean flag.
4. **One-Hot Encoding:** Converted high-cardinality categorical features (`Sex`, `Embarked`, `Title`) into machine-readable numeric binary vectors.

## 📊 Model Performance & Metrics

Evaluated strictly on an unseen 20% validation holdout set to mirror real-world generalization:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Validation Accuracy** | **84.00%** | High overall correctness on validation split |
| **Precision (Class 1 - Survived)** | **0.82** | Low false-positive rate when predicting survivors |
| **Recall (Class 1 - Survived)** | **0.78** | Effectively captures the majority of actual survivors |
| **F1-Score (Macro Average)** | **0.83** | Balanced harmonic mean across both classes |

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[Kaggle Dataset] --> B[train.csv]
        A --> C[test.csv]
    end

    subgraph Preprocessing & Feature Engineering
        B & C --> D[Title Extraction via Regex]
        D --> E[Grouped Median Age Imputation]
        E --> F[Family Dynamics & IsAlone Flags]
        F --> G[Categorical One-Hot Encoding]
    end

    subgraph Modeling Pipeline
        G --> H[80/20 Train & Validation Split]
        H --> I[Gradient Boosting Classifier]
        I --> J[Tree Depth & Learning Rate Tuning]
    end

    subgraph Evaluation & Output
        J --> K[Validation Metrics: F1, Recall, Precision]
        J --> L[Test Set Predictions]
        L --> M[submission.csv Generation]
    end

