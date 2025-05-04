from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Create Spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Task 1: Load & Explore
df = spark.read.csv("/workspaces/assignment-3-fake-news-detection-Umamaheshwareidigi/fake_news_sample.csv", header=True, inferSchema=True)
df = df.dropna(subset=["text", "label", "id", "title"])  # Drop incomplete rows
df.createOrReplaceTempView("news_data")
df.show(5)
print("Total articles:", df.count())
df.select("label").distinct().show()

# Task 2: Preprocess
df = df.withColumn("text_lower", lower(col("text")))
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
df_words = tokenizer.transform(df)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_words).select("id", "title", "filtered_words", "label")

# Task 3: Feature Extraction
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df_filtered)
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf).select("id", "title", "features", "label_index")

# Task 4: Train/Test Split (Check distribution!)
train, test = df_final.randomSplit([0.8, 0.2], seed=42)

print("Train Label Distribution:")
train.groupBy("label_index").count().show()

print("Test Label Distribution:")
test.groupBy("label_index").count().show()

# Train the model
lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=20)
model = lr.fit(train)

# Task 5: Prediction & Evaluation
predictions = model.transform(test).select("id", "title", "label_index", "prediction")

# Evaluation
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# Save metrics
pd.DataFrame([
    ["Accuracy", round(accuracy, 4)],
    ["F1 Score", round(f1_score, 4)]
], columns=["Metric", "Value"]).to_csv("task5_output.csv", index=False)

# Optional: Confusion Matrix Debugging
print("Confusion Matrix:")
predictions.groupBy("label_index", "prediction").count().show()

# Output
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
