# 🚀 LifecycleX — End-to-End Product Growth Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-orange)
![Experimentation](https://img.shields.io/badge/A%2FB%20Testing-Enabled-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌐 Live Demo
👉 https://growth-analysis-lifecyclex.streamlit.app/

---

## 🧭 TL;DR (Executive Summary)

LifecycleX is a **full-stack product analytics case study** analyzing a simulated product with **50,000 users across the full lifecycle**:

- Funnel → Activation → Conversion  
- A/B Testing → Experiment impact  
- Retention → Cohort decay  
- Revenue → ARPU & segmentation  
- Churn → Predictive modeling  

📌 **Core Insight:**
> Growth optimized for activation alone can degrade retention — sustainable growth requires optimizing for **LTV (Lifetime Value)**.

---

## 📌 Problem Statement

Most product teams optimize **top-of-funnel metrics (activation, conversion)**  
but fail to evaluate **downstream impact (retention, churn, revenue quality)**.

This project answers:

- Where is the biggest bottleneck in the funnel?
- Does increasing activation improve revenue?
- Is there a hidden trade-off with retention?
- Which users drive long-term value?
- Can churn be predicted early?

---

## 🧠 Key Results

| Metric | Value |
|------|------|
| Users | 50,000 |
| Activation Rate | 37.75% |
| Purchasers | 5,277 |
| Revenue | $392,280 |
| ARPU | $7.85 |
| ARPPU | $74.34 |
| Churn Model ROC-AUC | **0.864** |

---

## 🔍 Analysis Breakdown

### 🔻 1. Funnel Analysis

- Largest drop-off at **activation stage**
- Desktop users outperform mobile
- Power users have significantly higher engagement

👉 **Insight:** Activation is the highest leverage growth point

---

### 🧪 2. A/B Experiment

| Metric | Control | Treatment | Lift |
|------|--------|----------|------|
| Activation | 35.36% | 40.12% | +13.5% |
| Conversion | 10.04% | 11.06% | +1.02% |

- Statistically significant improvement (p < 0.001)

👉 **Insight:** Experiment improves top-of-funnel metrics

---

### 🔁 3. Retention Analysis

- Retention declines from **~93% → ~47% (Week 10)**
- Treatment shows **weaker early retention**

👉 **Critical Insight:**
> Increased activation introduces lower-quality users → retention trade-off

---

### 💰 4. Revenue Impact

| Metric | Control | Treatment |
|------|--------|----------|
| Revenue | $183K | $208K |
| ARPU | $7.37 | $8.32 |

- Revenue concentrated in **power users**
- Treatment increases both activation and ARPU

👉 **Insight:** Growth is driven by high-value segments

---

### ⚠️ 5. Churn Prediction

- Model: Logistic Regression  
- ROC-AUC: **0.864**

👉 Key drivers:
- Early engagement
- Device type
- User segment

👉 **Insight:**
> Churn can be predicted early → strong opportunity for intervention

---

## 🧪 Product Thinking & Recommendations

### 🎯 Segment-Aware Rollout
- Prioritize **power users + desktop**
- Avoid blanket rollout

---

### 🔄 Improve Early Retention (Weeks 1–2)
- Onboarding nudges
- Behavioral triggers
- Value reinforcement

---

### 📊 Optimize for LTV (Not Just Activation)

Instead of: Activation
Optimize: Activation × Retention × Revenue


---

### 📱 Mobile Optimization Needed
- Lower conversion rates
- Higher churn risk

---

## 🛠️ Tech Stack

- **Python** (Pandas, NumPy)
- **Machine Learning** (Scikit-learn)
- **Statistics** (Hypothesis Testing)
- **Visualization** (Matplotlib / Seaborn / Plotly)
- **Dashboarding** (Streamlit)

---
