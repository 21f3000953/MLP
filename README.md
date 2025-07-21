# Movie Review Sentiment Prediction

This project is a sentiment analysis task on movie reviews. It was implemented as part of a term-break assignment and aims to classify the sentiment of movie reviews using machine learning models.

## Dataset

The dataset comprises:
- train.csv: Training data with labeled sentiments.
- test.csv: Test data with unlabeled reviews.
- movies.csv: Metadata related to the movies (possibly for enrichment).
- sample.csv: A sample submission format for Kaggle.

## Objective

Predict the sentiment of movie reviews as a classification problem. The task involves:
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Generating predictions for test data

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Workflow

1. Import Libraries
   - All necessary data science libraries are imported: numpy, pandas, sklearn, matplotlib, seaborn.

2. Load and Inspect Data
   - Data is read from CSV files using Pandas.
   - Checks for null values and performs basic statistical description using .describe() and .isnull().

3. Exploratory Data Analysis (EDA)
   - Visualization of sentiment frequencies.
   - Possibly investigates the distribution of sentiments and reviews.

4. Preprocessing
   - Categorical encoding using LabelEncoder, OneHotEncoder, and OrdinalEncoder.
   - Scaling with StandardScaler and MinMaxScaler.

5. Model Training
   - Models used: SGDClassifier, RidgeClassifier, LogisticRegression.
   - Uses cross_val_predict and RandomizedSearchCV for tuning and evaluation.
   - Evaluated with metrics like precision, recall, confusion matrix, and classification report.

6. Prediction
   - Predictions are made on the test dataset using the trained model.
   - Output prepared in submission format.

## Results and Metrics

- The notebook includes metrics such as precision, recall, and confusion matrix to evaluate model performance.
- Visual tools like ConfusionMatrixDisplay and precision_recall_curve are used for performance analysis.

## How to Run

1. Clone the repository or download the notebook.
2. Ensure you have the required datasets (train.csv, test.csv, etc.) in the correct folder structure.
3. Install dependencies:

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

4. Run the notebook using Jupyter or any IDE that supports .ipynb.

## Folder Structure

```
.
├── train.csv
├── test.csv
├── movies.csv
├── sample.csv
├── 21f3000953-notebook-t22023.ipynb
└── README.md
```

## Author

Name: Shreya Garg  
Assignment: Term Break 1 — Sentiment Prediction on Movie Reviews  
Platform: Kaggle

## Notes

- This notebook uses traditional ML models rather than deep learning or NLP techniques like LSTM or Transformers.
- Label encoding and standard ML preprocessing are effectively applied.
- Could be further improved by including NLP-based features like TF-IDF or word embeddings.
