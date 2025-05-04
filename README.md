# Assignment-5-FakeNews-Detection

# Fake News Detection using PySpark

This project implements a machine learning pipeline in PySpark to classify news articles as **real** or **fake** based on their textual content.

## 📂 Dataset

The input dataset is a CSV file named `fake_news_sample.csv` containing the following columns:

- `id`: Unique identifier for each article.
- `title`: Title of the news article.
- `text`: Full text of the article.
- `label`: Ground truth label (`FAKE` or `REAL`).


---

## 🚀 Pipeline Overview

### ✅ Step 1: Load & Explore
- Load data into Spark DataFrame.
- Drop rows with null values in critical columns.
- Display first 5 rows and unique label values.

### 🔄 Step 2: Text Preprocessing
- Convert text to lowercase.
- Tokenize the text into words.
- Remove English stopwords.

### 🧠 Step 3: Feature Extraction
- Apply HashingTF to get term frequencies.
- Apply IDF to get TF-IDF features.
- Encode string labels using `StringIndexer`.

### 🏋️ Step 4: Model Training
- Split data into training and test sets (70:30).
- Train a `LogisticRegression` classifier.

### 📈 Step 5: Evaluation
- Predict labels on test set.
- Evaluate using **Accuracy** and **F1 Score**.
- Display confusion matrix.

---

## 📊 Sample Outputs

Output CSV files:
- `task1_output.csv`: First 5 rows of original data.
- `task2_output.csv`: Tokenized and cleaned text.
- `task3_output.csv`: TF-IDF features and label index.
- `task4_output.csv`: Predictions on the test set.
- `task5_output.csv`: Accuracy and F1 score.

---

## 📝 How to Run

1. Ensure PySpark and Pandas are installed:
   ```bash
   pip install pyspark pandas faker
   ```

2. To generate the input csv file
```bash
python Dataset_Generator.py
```
3.To run the task
```bash
python task.py
```   


