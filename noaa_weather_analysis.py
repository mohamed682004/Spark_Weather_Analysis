import sys
import os
import time

# Check if findspark is needed (helpful if SPARK_HOME isn't in PATH)
try:
    import findspark
    findspark.init('/opt/spark')  # Adjust this path if your Spark is installed elsewhere
except ImportError:
    pass # If findspark isn't installed, we assume Spark is already in the environment variables

import plotext as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, dayofyear, corr, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# --- CONFIGURATION ---
PARQUET_FOLDER = "ghcn_2024.parquet"

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# --- HELPER: STATUS LOGGER ---
def log_status(message, stage=None):
    """Prints a professional status update with timestamps."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    if stage:
        print(f"\n[{timestamp}] === PHASE: {stage} ===")
        print(f"[{timestamp}] ℹ️  {message}")
    else:
        print(f"[{timestamp}] ℹ️  {message}")

# --- STEP 1: INITIALIZE ENVIRONMENT ---
log_status("Booting up Spark on Zorin OS...", stage="INIT")
t_start = time.time()

# OPTIMIZATION: Reduced memory to 2g to prevent OS killing the process
spark = SparkSession.builder \
    .appName("NOAA_Pro_Stats") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
log_status("Spark Session Active. JVM Heap Initialized.")

# --- STEP 2: DATA LOADING ---
log_status("Reading Parquet Data...", stage="LOAD")

if not os.path.exists(PARQUET_FOLDER):
    print("!!! ERROR: Parquet file missing. Run the first script to download data.")
    sys.exit(1)

df = spark.read.parquet(PARQUET_FOLDER)
log_status(f"Raw data schema loaded. Columns: {df.columns}")

# --- STEP 3: PREPARATION & FEATURE ENGINEERING ---
log_status("Pivoting and Generating Features...", stage="PREP")

# We clean, pivot, and ADD new features (Month, DayOfYear) to test dependencies
clean_df = df.filter((col("q_flag").isNull()) & (col("element").isin("TMAX", "TMIN")))

pivot_df = clean_df \
    .withColumn("date", to_date(col("date_int"), "yyyyMMdd")) \
    .withColumn("temp_c", col("value") / 10.0) \
    .groupBy("station_id", "date") \
    .pivot("element") \
    .avg("temp_c") \
    .na.drop() \
    .withColumn("Month", month("date")) \
    .withColumn("DayOfYear", dayofyear("date"))

# Cache data in RAM because we will run multiple tests on it
pivot_df.cache()
row_count = pivot_df.count()
log_status(f"Dataset ready. Cached {row_count:,} rows in memory.")

# --- STEP 4: STATISTICAL DEPENDENCY TESTS ---
log_status("Running Statistical Dependency Tests...", stage="STATS")

# A. Prepare Vector for Correlation Matrix
# We want to see dependencies between: TMAX, TMIN, Month, DayOfYear
feature_cols = ["TMAX", "TMIN", "Month", "DayOfYear"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
vector_df = assembler.transform(pivot_df)

# B. Pearson Correlation (Linear Relationship)
log_status("Calculating Pearson Correlation (Linear Dependency)...")
pearson_result = Correlation.corr(vector_df, "features", "pearson").head()
p_matrix = pearson_result[0].toArray().tolist() # Convert DenseMatrix to Python List

# C. Spearman Correlation (Rank Relationship - Non-Linear)
log_status("Calculating Spearman Correlation (Monotonic Dependency)...")
spearman_result = Correlation.corr(vector_df, "features", "spearman").head()
s_matrix = spearman_result[0].toArray().tolist()

log_status("Statistical calculations complete.")

# --- STEP 5: VISUALIZATIONS ---
log_status("Rendering Terminal Visuals...", stage="VISUALS")

# VISUAL 1: THE HEATMAP (Pearson)
print("\n" + "="*40)
print(" VISUAL 1: FEATURE CORRELATION HEATMAP")
print("="*40)
print("Legend: Red = Strong Positive Dependency, Blue = Strong Negative, Green = None")

# Plotext Matrix Plot requires a list of lists
plt.clear_data()
plt.theme("dark")
plt.matrix_plot(p_matrix)
plt.title("Pearson Correlation Matrix")
plt.xticks(list(range(4)), feature_cols)
plt.yticks(list(range(4)), feature_cols)
plt.show()

# VISUAL 2: SCATTER PLOT (Direct Dependency)
print("\n" + "="*40)
print(" VISUAL 2: TMIN vs TMAX DEPENDENCY")
print("="*40)

# OPTIMIZATION: Added limit(5000) to prevent collecting too much data and crashing JVM
sample_data = pivot_df.select("TMIN", "TMAX").sample(0.1, seed=42).limit(5000).collect()
x_val = [r['TMIN'] for r in sample_data]
y_val = [r['TMAX'] for r in sample_data]

plt.clear_data()
plt.scatter(x_val, y_val, marker="dot", color="yellow")
plt.title("Dependency Check: Min vs Max Temp")
plt.xlabel("Min Temp (C)")
plt.ylabel("Max Temp (C)")
plt.show()

# --- PART 4: TERMINAL VISUALIZATION (Continued) ---
print_section("STEP 4 (Cont): Additional Visuals")

# VISUAL 3: HISTOGRAM
print(">>> Graph 3: TMAX Distribution Histogram (Sampled)")
# OPTIMIZATION: Added limit(5000) to ensure safety
tmax_sample = pivot_df.select("TMAX").sample(withReplacement=False, fraction=0.1).limit(5000).rdd.flatMap(lambda x: x).collect()

plt.clear_data() 
plt.hist(tmax_sample, bins=30, color="blue", label="TMAX Freq")
plt.title("Distribution of Max Temperatures (2024)")
plt.xlabel("Temperature (C)")
plt.show()

# VISUAL 4: SEASONAL TREND
print("\n>>> Graph 4: Average Temperature by Month")

# Aggregate in Spark FIRST (Small data), then bring to Python (Driver)
monthly_data = pivot_df.withColumn("Month", month("date")) \
    .groupBy("Month") \
    .agg(avg("TMAX").alias("Avg_TMAX"), avg("TMIN").alias("Avg_TMIN")) \
    .orderBy("Month") \
    .collect()

# Prepare lists for plotting
months = [row['Month'] for row in monthly_data]
tmax_vals = [row['Avg_TMAX'] for row in monthly_data]
tmin_vals = [row['Avg_TMIN'] for row in monthly_data]

plt.clear_data()
plt.theme("dark") 
plt.plot(months, tmax_vals, label="Avg Max Temp", marker="dot", color="red")
plt.plot(months, tmin_vals, label="Avg Min Temp", marker="dot", color="cyan")
plt.title("Seasonal Temperature Trend")
plt.xlabel("Month (1-12)")
plt.ylabel("Temp (C)")
plt.show()

# --- STEP 6: TEXT REPORT ---
log_status("Generating Analysis Report...", stage="REPORT")

print("\n--- STATISTICAL DEPENDENCY REPORT ---")
print(f"1. TMIN vs TMAX Correlation: {p_matrix[0][1]:.4f}")
if p_matrix[0][1] > 0.8:
    print("   -> CONCLUSION: High Dependency. If TMIN goes up, TMAX almost certainly goes up.")
elif p_matrix[0][1] > 0.5:
    print("   -> CONCLUSION: Moderate Dependency.")

print(f"2. Month vs Temperature Correlation: {p_matrix[0][2]:.4f}")
print("   -> NOTE: This is usually low because temperature is cyclic (High in summer, low in winter).")
print("      Linear correlation fails to capture waves/cycles effectively.")

t_end = time.time()
print("\n" + "="*50)
log_status(f"Job Complete. Total Runtime: {t_end - t_start:.2f} seconds.")
print("="*50)

spark.stop()
