# u_of_t_dsi_disease_prediction_ml_project
Repository to store and showcase the work done for the final project

**Author**: Shafayat Syed

**Program**: U of T Data Sciences Institute Machine Learning Program (Cohort 6)

**Project**: Disease prediction using ML

**Project objective**

The business question that we had set out to answer was 

Classification: Which symptom features are the most significant predictors of a disease (eg., Allergy)?

- For our case we targeted "Heart attack". We want to understand which symptoms are most significant predictors of "Heart attack"

**Final outcome of the project:**

- We found four key takeaways:
    - Symptoms like "Chest pain", "Sweating", "Breathlessness" and "Vomiting" are significant predictors of a patient having a "Heart attack"
    - However, it is not a good idea to just go by these direct symptoms because these symptoms are partially shared by three other prognosis (GERD, Pneumonia and Tuberculosis)
    - Hence, a physician, a healthcare worker or an individual must try to isolate/exclude the following symptoms (strongly associated with GERD, Pneumonia and Tuberculosis) to accurately identify a potential heart attack prognosis
        - acidity, 
        - blood_in_sputum, 
        - chills, 
        - cough, 
        - fatigue, 
        - fever, 
        - malaise, 
        - loss_of_apetite, 
        - phelgm, 
        - rusty_sputum, 
        - stomach_pain, 
        - ulcers_on_tongue, 
        - swelled_lymph_nodes, 
        - weightloss
        - yellowing_of_eyes 

    - Due to symptom separation in the dataset we got unusual accuracy, which is almost impossible in the real-world. Hence, we may need to find other datasets to be able to predict these diseases with accuracy. This includes not just binary (yes/no) symptoms, but also the gradience of it in numeric values. We imagine chest pain in GERD will be significantly less painful than that experienced by someone having a Heart attack. Hence, just binary symptoms may not be enough or prudent to use in an ML application context.

**Data Source**
Kaggle, Disease Prediction using Machine Learning 
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning


**Methodology**

- This research developed a machine learning model for predicting "Heart attack" from the symptoms
- While considering "Heart attack" we reviewed three prognosis (Tuberculosis, Pneumonia, and GERD) based on the cosine similarity of symptoms
- Data preparation involved filtering for target prognoses, and dimensionality reduction, and strictly preventing data leakage through train and test split
- Initial analysis explored symptom similarity and prevalence patterns to better understand the relationship of symptoms and prognosis
- A Random Forest Classifier was chosen for its balance of performance and interpretability
- The model was robustly trained and evaluated using 5-fold Stratified Cross-Validation
- Hyperparameter optimization was performed using GridSearchCV
- Consistent 100% accuracy across Random Forest, Logistic Regression, and Decision Tree models was observed
- The unusually high and in fact impractical in real world accuracy demonstrates how the dataset was encoded and the dataset's unique symptom-prognosis relationships
    - The high accuracy was investigated with Decision Tree model which confirmed that the perfect performance is due to symptoms being mutually exclusive within the dataset
- This indicates inherent data separability within the dataset and our approach to validating our methodology and ruling out common issues


**Consideration of the guiding questions**

Reviewing Dataset

What are the key variables and attributes in your dataset?

- Complete Dataset consists of 2 CSV files. One of them is training and other is for testing your model
- Each CSV file has 133 columns. 132 of these columns are symptoms that a person experiences and last column is the prognosis
- The 'prognosis' column is ou target value for prediction
- The 132 symptoms are mapped to 42 diseases or prognosis

How can we explore the relationships between different variables?
- This is a binary tagging of symptoms. So, looking at the mean occurence and prevanlence of symptoms across prognosis is a good exploration idea.
- We also looked at chi-squared tests for significance.
    - The Chi-squared test of independence is used to determine if there is a statistically significant relationship between two categorical variables. In this case, we were testing each binary symptom (categorical: 0 or 1) against the categorical prognosis variable.

Are there any patterns or trends in the data that we can identify?
- We can see that four prognosis share some of the similar symptoms (one or more), which could lead to confusion.

Who is the intended audience?
- This analysis is catered for Health Data Analysts, ML Practitioners and the outcome can be read by medical practitioners with the biases and limitations explained below.

What is the question our analysis is trying to answer?
- We want to understand which symptoms are most significant predictors of "Heart attack", and if some prognosis have similar symptoms then what can be some isolating or guiding features for diagnosis.

Are there any specific libraries or frameworks that are well-suited to our project requirements?
- numpy, scikit-learn, matplotlib and seaborn

Data Visualization Guiding Questions

What are the main goals and objectives of our visualization project?
- We must be able to show prognoses which are similar or dissimilar 
- We must then be able to see symptoms which are common or uncommon among the prognoses 
- We must also be able to explain which symptoms are key to making decisions (explainability)

How can we tailor the visualization to effectively communicate with our audience?
- We must be able to show a lot of information in one chart
- We must also be able to compare and contrast among the four key prognosis

What type of visualization best suits our data and objectives (e.g., bar chart, scatter plot, heatmap)?
- Barplots
- Heatmaps

How can we iterate on our design to address feedback and make iterative improvements?
- There is room for improvement for the visualization and design choices, especially fine-tune the charts with user feedback.

What best practices can we follow to promote inclusivity and diversity in our visualization design?
- We will try to make the charts as simple and readable as possible. 

How can we ensure that our visualization accurately represents the underlying data without misleading or misinterpreting information?
- This will require user-feedback but we did show it to less technical individuals to obtain their input on this. We also consulted a physician.

Are there any privacy concerns or sensitive information that need to be addressed in our visualization?
- None, this is an anonimized dataset. 

Machine Learning Model Guiding Questions

What are the specific objectives and success criteria for our machine learning (ML) model?
- The objective of the ML Model is to be able to predict "Heart attack" accurately, but also be able to predict it when it has prognoses which share similar symptoms.

How can we select the most relevant features for training our machine learning model?
- We removed the columns that had 0 symptom occurrence across all training records and we excluded it from the testing set too. This reduced a lot of redundant columns from our dataset. However, in the real world since a patient can come with more than one symptoms and could have more than one prognosis, hence we chose to keep the symptom occurence that had at least one presence across the test set. 

Are there any missing values or outliers that need to be addressed through preprocessing?
- None.

Which machine learning algorithms are suitable for our problem domain?
- Since, this is a binary data across symptoms for the whole dataset. We identified logistic regression (common for this type of problem, simple as well), decision-tree and random forest being good candidates given the nature of our symptom data and the patterns you've observed. 

Here is why:

***Random Forest Classifier*** (Preferred Candidate Model)
Ensemble Power & Robustness: Random Forest builds many independent Decision Trees (a "forest") and combines their predictions. This ensemble approach significantly enhances robustness, reduces the risk of overfitting (compared to a single, deep Decision Tree), and generally leads to higher predictive accuracy.

Handles Feature Interactions: Like individual Decision Trees, Random Forests are excellent at discovering and utilizing complex interactions between multiple symptoms, even if those interactions aren't immediately obvious.

Reliable Feature Importance: Random Forests provide highly reliable feature importances. These scores indicate which symptoms were most consistently and significantly used by the ensemble of trees to make accurate classifications. This directly supports your objective of identifying the most differentiating and "isolating" symptoms.

Confirmation of Data Quality: Similar to the Decision Tree, the 100% accuracy with a Random Forest further confirms the exceptional separability of your dataset, as even a robust ensemble model finds no ambiguity.

***Decision Tree Classifier*** (Partner Model for Explanation)
Direct Rule Extraction: Decision Trees inherently learn a series of "if-else" rules based on your symptoms. For binary (0/1) symptom data, these rules are extremely intuitive (e.g., "IF 'chest_pain' is 1 AND 'cough' is 0, THEN diagnosis is X"). This directly aligns with the goal of identifying clear, isolating symptoms.

Handles Non-Linearity: Unlike linear models (like Logistic Regression), Decision Trees can capture complex, non-linear relationships and interactions between symptoms without requiring explicit feature engineering for these interactions.

Interpretability: When a Decision Tree is not excessively deep, its structure can be directly visualized, providing a transparent view of the exact diagnostic logic the model employs. This is invaluable for understanding why a particular prediction is made.

Confirmation of Separability: The fact that a Decision Tree achieved 100% accuracy is the strongest evidence of your dataset's perfect separability. It means there is a sequence of simple, unambiguous symptom checks that can perfectly distinguish each prognosis.

What techniques can we use to validate and tune the hyperparameters for our models?
- We use cross-validation (e.g., K-Fold or Stratified K-Fold) for robust validation, providing reliable performance estimates. For tuning, GridSearchCV systematically searches a predefined hyperparameter space, using cross-validation to find the optimal combination that maximizes a chosen scoring metric.

How should we split the dataset into training, validation, and test sets?
- We did not use the train and test set as provided by the dataset owner because it only has one sample of each prognosis in the test data. Instead, we created a combined dataset and then we split it into train and test.

- We performed an initial split to create a final, unseen test set (e.g., 80% train/validation, 20% test). Then, use cross-validation on the training/validation portion for model training and hyperparameter tuning for the Random Forest model. The final test set is used only once, at the very end, for an unbiased performance evaluation.

Are there any ethical implications or biases associated with our machine learning model?

Yes, potential biases and limitations exist. 

- If the training data disproportionately represents certain demographics or symptom presentations, the model may perform poorly or inaccurately for underrepresented groups. 
- Achieving 100% accuracy also raises concerns about real-world generalizability from this data, as it might fail on diverse, noisy patient data, leading to misdiagnoses.
- The dataset must be investigated further for accuracy, timeliness, appropriateness and representation before insights from this model can be used in the real-world.
- Hence, this model for now should be used for learning purposes only.

How can we document our machine learning pipeline and model architecture for future reference?
- We have left comments on our code
- We have left our reflections, comments and analysis at the end of the notebooks
- We are including detailed information on this readme
- We will try to save the model as a pickle (Pkl) file 