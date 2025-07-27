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

   For a full list of prognosis similar to heart attack, please see [this visualization](https://github.com/Robocoph/u_of_t_dsi_disease_prediction_ml_project/blob/main/visualization/01_heatmap_prognosis_similar_to_cancer.png)

2. **Differential Diagnosis Requirements**: To accurately identify heart attack, healthcare providers should consider **excluding** the following symptoms strongly associated with other conditions:

   ```
   • acidity              • blood_in_sputum       • chills
   • cough                • fatigue               • fever
   • malaise              • loss_of_appetite      • phlegm
   • rusty_sputum         • stomach_pain          • ulcers_on_tongue
   • swelled_lymph_nodes  • weight_loss           • yellowing_of_eyes
   ```

     For a full list of symptoms, please see [this visualization](https://github.com/Robocoph/u_of_t_dsi_disease_prediction_ml_project/blob/main/visualization/04_heatmap_symptom_prevalence_across_difference_prognosis.png)

3. **Dataset Limitations**: The unusually high accuracy (100%) indicates perfect symptom separation in the dataset, which is unrealistic in real-world scenarios. This suggests the need for:
   - A larger dataset with more diverse set of symptom representation
   - Gradient-based symptom severity (not just binary yes/no)
   - Quantitative scales and symptom intensity measures
   - Women and men also suffer different symptoms during Heart attack, so demographic information would be valuable to consider

   The confusion matrix can be found [here](https://github.com/Robocoph/u_of_t_dsi_disease_prediction_ml_project/blob/main/visualization/09_confusion_matrix_random_forest_classifier.png)

## 📊 Data Source

**Dataset**: [Disease Prediction using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Platform**: Kaggle
- **Structure**: 2 CSV files (training and testing)
- **Features**: 132 binary symptom columns + 1 prognosis column
- **Coverage**: 42 different diseases/prognoses

## 🔬 Methodology

### Data Preparation
- Cleaned prognosis data for whitespace
- Implemented strict train-test split to prevent data leakage
- Reduce dimensions on training dataset
- Applied cosine similarity analysis for symptom relationship exploration for prognoses similar to Heart attack
- Filtered dataset for three closest prognoses and the target prognosis (Heart attack, GERD, Pneumonia, Tuberculosis)
- Performed dimensionality reduction by removing columns that had all 0 values or negative symptom response in the training set
   - Once removed from training set, it was removed from testing set too before it was fit on the model


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

For all visualizations, please check out the [visualization folder](https://github.com/Robocoph/u_of_t_dsi_disease_prediction_ml_project/tree/main/visualization)

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
- Diverse symptom to prognosis will be required, especially considering symptom diversity in gender for heart attack
- Integration with clinical expertise essential
- Monitoring for bias and accuracy needed with additional demographic information
- Patient safety protocols must be prioritized first before implementing any ML solution in healthcare

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
├── visualizations/             # Plots, charts and graphs important for analysis
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
- **Model Persistence**: (Coming soon!) models in pickle format for reproducibility

## 🤝 Contributing

This project was developed as part of an academic program. For questions or collaboration opportunities, please reach out through the contact information provided.

---

## 📊 Future Work

- [ ] Jurisdictional scan to investigate and search for larger, more diverse datasets  
- [ ] Match with real-world clinical validation studies
- [ ] Consultation with clinicians for improvements to our assumptions, analysis and insights
- [ ] Exported encodings and models in pickle format for reproducibility

---

**📧 Contact**: https://www.linkedin.com/in/shafayat-syed/
**🎓 Institution**: University of Toronto Data Sciences Institute  
**📅 Completion Date**: 2025-07-27