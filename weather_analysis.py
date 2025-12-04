import findspark
findspark.init('/opt/spark')  # Explicitly point to your Spark installation
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, month, year, count, when, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# --- STEP 1: Initialize Spark Session ---
spark = SparkSession.builder \
    .appName("WeatherAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN") # Reduce console clutter

print(">>> Spark Session Started!")

# --- STEP 2: Load Data ---
file_path = "weather.parquet"  # Make sure this matches your file location
df = spark.read.parquet(file_path)

print(f">>> Data Loaded. Total Rows: {df.count()}")
df.printSchema()
df.show(5)

# --- STEP 3: Data Cleaning ---
# Create derived columns that might be useful
# Let's use average temperature and average humidity for analysis
df = df.withColumn("AvgTemp", (col("MinTemp") + col("MaxTemp")) / 2)
df = df.withColumn("AvgHumidity", (col("Humidity9am") + col("Humidity3pm")) / 2)
df = df.withColumn("AvgWindSpeed", (col("WindSpeed9am") + col("WindSpeed3pm")) / 2)

# 1. Drop rows with missing values in critical columns
clean_df = df.na.drop(subset=["AvgTemp", "AvgHumidity", "AvgWindSpeed", "Rainfall"])

# 2. Filter out impossible data (e.g., Humidity > 100%)
clean_df = clean_df.filter((col("AvgHumidity") <= 100) & (col("AvgHumidity") >= 0))

print(f">>> Data Cleaned. Rows remaining: {clean_df.count()}")

# --- STEP 4: Statistical Analysis ---
# What is the average temperature per month?
print(">>> Calculating Average Temperature per Month...")
# First, we need to extract month from a date column
# If you don't have a date column, you might need to use an index or create one
# For this example, let's assume we'll analyze by available data
# Or you can add a dummy date column if needed

# Let's do basic statistics instead:
print(">>> Basic Statistics:")
clean_df.select("AvgTemp", "AvgHumidity", "AvgWindSpeed", "Rainfall").describe().show()

# --- STEP 5: Machine Learning (Linear Regression) ---
# Goal: Predict 'Rainfall' based on 'AvgTemp', 'AvgHumidity', and 'AvgWindSpeed'

# A. Feature Engineering
assembler = VectorAssembler(
    inputCols=["AvgTemp", "AvgHumidity", "AvgWindSpeed"], 
    outputCol="features"
)

ml_data = assembler.transform(clean_df)
# Select only the columns we need for the model
final_data = ml_data.select("features", "Rainfall")

# B. Split Data (80% Training, 20% Testing)
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

# C. Train Model
lr = LinearRegression(labelCol="Rainfall", featuresCol="features")
lr_model = lr.fit(train_data)

print(f">>> Model Trained!")
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# D. Evaluate Model
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="Rainfall", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f">>> Root Mean Squared Error (RMSE): {rmse}")
print("Sample Predictions:")
predictions.select("Rainfall", "prediction").show(10)

# Additional: Show correlation between predicted and actual
print(">>> Prediction Statistics:")
predictions.select("Rainfall", "prediction").describe().show()

# --- Optional: Predict RainTomorrow (Classification Problem) ---
# If you want to predict RainTomorrow (Yes/No), we could use logistic regression
print("\n>>> Optional: Creating binary label for RainTomorrow")
# Create binary label (1 for Yes, 0 for No)
binary_df = clean_df.withColumn("RainTomorrowBinary", 
                                 when(col("RainTomorrow") == "Yes", 1).otherwise(0))
binary_df.select("RainTomorrow", "RainTomorrowBinary").show(5)

# --- Stop Session ---
spark.stop()
print(">>> Analysis Complete!")
