# 🏥 Disease Prediction ML Project
> **Heart Attack Prediction using Machine Learning**

A comprehensive machine learning project focused on predicting heart attacks from symptom data, developed as part of the University of Toronto Data Sciences Institute Machine Learning Program.

---

## 📋 Project Overview

| **Field** | **Details** |
|-----------|-------------|
| **Author** | Shafayat Syed |
| **Program** | U of T Data Sciences Institute ML Program (Cohort 6) |
| **Focus** | Disease prediction using machine learning |
| **Target Disease** | Heart Attack |

## 🎯 Project Objective

**Primary Research Question:** 
> Which symptom features are the most significant predictors of a heart attack?

This project aims to identify key symptoms that can help healthcare professionals and individuals recognize potential heart attack cases while understanding the complexity of symptom overlap with other conditions.

## 🔍 Key Findings

### 🚨 Primary Heart Attack Indicators
The following symptoms emerged as **significant predictors** of heart attack:
- **Chest pain**
- **Sweating** 
- **Breathlessness**
- **Vomiting**

### ⚠️ Critical Considerations

1. **Symptom Overlap Challenge**: These primary symptoms are partially shared with three other conditions:
   - GERD (Gastroesophageal Reflux Disease)
   - Pneumonia
   - Tuberculosis

2. **Differential Diagnosis Requirements**: To accurately identify heart attack, healthcare providers should consider **excluding** the following symptoms strongly associated with other conditions:

   ```
   • acidity              • blood_in_sputum       • chills
   • cough                • fatigue               • fever
   • malaise              • loss_of_appetite      • phlegm
   • rusty_sputum         • stomach_pain          • ulcers_on_tongue
   • swelled_lymph_nodes  • weight_loss           • yellowing_of_eyes
   ```

3. **Dataset Limitations**: The unusually high accuracy (100%) indicates perfect symptom separation in the dataset, which is unrealistic in real-world scenarios. This suggests the need for:
   - More comprehensive datasets
   - Gradient-based symptom severity (not just binary yes/no)
   - Quantitative scales and symptom intensity measures
   - Women and men also suffer different symptoms during Heart attack, so demographic information should be considered too

## 📊 Data Source

**Dataset**: [Disease Prediction using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Platform**: Kaggle
- **Structure**: 2 CSV files (training and testing)
- **Features**: 132 binary symptom columns + 1 prognosis column
- **Coverage**: 42 different diseases/prognoses

## 🔬 Methodology

### Data Preparation
- Filtered dataset for target prognoses (Heart attack, GERD, Pneumonia, Tuberculosis)
- Implemented strict train-test split to prevent data leakage
- Performed dimensionality reduction by removing columns that had all 0 values or negative response in the training set
- Applied cosine similarity analysis for symptom relationship exploration

### Model Selection & Training
- **Primary Model**: Random Forest Classifier
- **Rationale**: Optimal balance of performance and interpretability
- **Validation**: 5-fold Stratified Cross-Validation
- **Optimization**: GridSearchCV for hyperparameter tuning

### Performance Analysis
- **Accuracy**: 100% (across Random Forest, Logistic Regression, and Decision Tree)
- **Investigation**: Used Decision Tree visualization to confirm perfect symptom separability
- **Conclusion**: High accuracy due to mutually exclusive symptom patterns in dataset

## 🛠️ Technical Stack

```python
# Core Libraries
numpy          # Numerical computations
scikit-learn   # Machine learning algorithms
matplotlib     # Data visualization
seaborn        # Statistical visualization
```

## 📈 Visualization Strategy

### Primary Objectives
- **Similarity Analysis**: Display prognoses relationships
- **Symptom Comparison**: Highlight unique symptoms for each prognosis
- **Model Explainability**: Provide clear decision-making insights

### Visualization Types
- **Bar Plots**: Symptom prevalence across conditions
- **Heatmaps**: Symptom-disease correlation matrices
- **Decision Trees**: Model decision pathways

## 🎯 Model Performance

### Algorithm Comparison
| Algorithm | Accuracy | Key Strengths |
|-----------|----------|---------------|
| **Random Forest** | 100% | Robust ensemble, reliable feature importance |
| **Decision Tree** | 100% | High interpretability, clear rules |
| **Logistic Regression** | 100% | Simple, fast, good baseline |

### Cross-Validation Results
- **Method**: 5-fold Stratified Cross-Validation
- **Consistency**: Perfect scores across all folds
- **Reliability**: Confirmed through multiple validation strategies

## ⚖️ Ethical Considerations & Limitations

### 🚨 Important Disclaimers

> **⚠️ FOR EDUCATIONAL USE ONLY**
> 
> This model should **NOT** be used for actual medical diagnosis without proper validation and clinical oversight.

### Identified Biases & Limitations

1. **Demographic Representation**: Potential underrepresentation of certain patient groups (especially, women)
2. **Real-world Generalizability**: Perfect accuracy suggests limited applicability to noisy clinical data
3. **Symptom Complexity**: Binary classification oversimplifies symptom gradients
4. **Dataset Quality**: Requires further validation for clinical accuracy and timeliness

### Recommendations for Clinical Application
- Comprehensive dataset validation required
- Integration with clinical expertise essential
- Continuous monitoring for bias and accuracy needed
- Patient safety protocols must be prioritized

## 🎓 Target Audience

- **Primary**: Health Data Analysts, ML Practitioners
- **Secondary**: Medical Practitioners (with appropriate disclaimers)
- **Academic**: Students and researchers in health informatics

## 📁 Repository Structure

```
u_of_t_dsi_disease_prediction_ml_project/
├── data/                       # Dataset files
├── data_exploration/           # Notebook for Data Exploration Work
├── decision_tree_model/        # Notebook for Decision Tree Model
├── random_forest_model/        # Notebook for Random Forest Model
├── logistic_regression_model/  # Notebook for Logistic Regression Model
├── Visualizations/             # Charts and graphs important for analysis
├── pkl_files/                  # WIP for .pkl files
└── README.md                   # This file
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python src/main_analysis.py
```

## 📝 Documentation

- **Code Comments**: Comprehensive inline documentation
- **Notebook Reflections**: Detailed analysis and insights
- **Model Persistence**: Saved models in pickle format for reproducibility

## 🤝 Contributing

This project was developed as part of an academic program. For questions or collaboration opportunities, please reach out through the contact information provided.

---

## 📊 Future Work

- [ ] Integration of gradient-based symptom severity
- [ ] Expansion to larger, more diverse datasets  
- [ ] Real-world clinical validation studies
- [ ] Development of uncertainty quantification methods
- [ ] Integration with electronic health records

---

**📧 Contact**: shafayat.syed@outlook.com 
**🎓 Institution**: University of Toronto Data Sciences Institute  
**📅 Completion Date**: 2025-07-27