import findspark
findspark.init('/opt/spark')  # Explicitly point to your Spark installation

import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, to_date

# --- CONFIGURATION ---
CSV_FILE = "2024.csv.gz"
PARQUET_FOLDER = "ghcn_2024.parquet"
ORC_FOLDER = "ghcn_2024.orc"

# --- INIT SPARK ---
print(">>> [INIT] Starting Spark Engine...")
spark = SparkSession.builder \
    .appName("Format_Battle_Royale") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# --- STEP 1: PREPARE DATA ---
print("\n>>> [PREP] Checking File Formats...")

# A. Check CSV
if not os.path.exists(CSV_FILE):
    print(f"!!! Error: {CSV_FILE} missing. Run previous scripts to download.")
    sys.exit(1)

# B. Check Parquet (Create if missing)
if not os.path.exists(PARQUET_FOLDER):
    print("... Generating Parquet file from CSV (One time setup)...")
    # Define Schema for CSV reading
    schema = StructType([
        StructField("station_id", StringType(), True),
        StructField("date_int", StringType(), True),
        StructField("element", StringType(), True),
        StructField("value", IntegerType(), True),
        StructField("m_flag", StringType(), True),
        StructField("q_flag", StringType(), True),
        StructField("s_flag", StringType(), True),
        StructField("obs_time", StringType(), True)
    ])
    df = spark.read.csv(CSV_FILE, schema=schema)
    df.write.parquet(PARQUET_FOLDER)

# C. Check ORC (Create if missing)
if not os.path.exists(ORC_FOLDER):
    print("... Generating ORC file from Parquet (This is the 'Transform' step)...")
    # It is faster to convert Parquet -> ORC than CSV -> ORC
    df_p = spark.read.parquet(PARQUET_FOLDER)
    df_p.write.orc(ORC_FOLDER)
    print("... ORC Generation Complete.")

# --- STEP 2: THE BENCHMARK FUNCTION ---
def run_benchmark(file_format, file_path, schema=None):
    """
    Loads data, filters, pivots, and counts.
    Returns: Time taken in seconds.
    """
    print(f"\n--- TESTING FORMAT: {file_format.upper()} ---")
    
    # Clear Cache to ensure a fair "Cold Start" for every test
    spark.catalog.clearCache()
    start_time = time.time()
    
    # 1. LOAD
    if file_format == "csv":
        df = spark.read.csv(file_path, schema=schema)
    elif file_format == "parquet":
        df = spark.read.parquet(file_path)
    elif file_format == "orc":
        df = spark.read.orc(file_path)

    # 2. TRANSFORM (Filter & Pivot)
    # We do a heavy operation to stress-test the format
    res = df.filter((col("q_flag").isNull()) & (col("element").isin("TMAX", "TMIN"))) \
            .withColumn("temp_c", col("value") / 10.0) \
            .groupBy("station_id", "date_int") \
            .pivot("element") \
            .avg("temp_c")
    
    # 3. ACTION (Force execution)
    count = res.count()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"   -> Rows Processed: {count:,}")
    print(f"   -> Time Taken: {duration:.2f} seconds")
    return duration

# --- STEP 3: EXECUTE BATTLE ---

# Define Schema again for CSV fairness
csv_schema = StructType([
    StructField("station_id", StringType(), True),
    StructField("date_int", StringType(), True),
    StructField("element", StringType(), True),
    StructField("value", IntegerType(), True),
    StructField("m_flag", StringType(), True),
    StructField("q_flag", StringType(), True),
    StructField("s_flag", StringType(), True),
    StructField("obs_time", StringType(), True)
])

# RUN TESTS
t_csv = run_benchmark("csv", CSV_FILE, csv_schema)
t_parquet = run_benchmark("parquet", PARQUET_FOLDER)
t_orc = run_benchmark("orc", ORC_FOLDER)

# --- STEP 4: RESULTS TABLE ---
print("\n" + "="*60)
print("FINAL RESULTS: STORAGE FORMAT SHOWDOWN")
print("="*60)
print(f"{'FORMAT':<10} | {'TIME (Sec)':<12} | {'SPEEDUP VS CSV':<15}")
print("-" * 45)

print(f"{'CSV':<10} | {t_csv:<12.2f} | {'1.0x (Baseline)':<15}")
print(f"{'ORC':<10} | {t_orc:<12.2f} | {t_csv/t_orc:<15.2f}")
print(f"{'PARQUET':<10} | {t_parquet:<12.2f} | {t_csv/t_parquet:<15.2f}")
print("-" * 45)

# --- VISUAL BAR CHART ---
def print_bar(label, value, max_val):
    bar_len = int((value / max_val) * 40)
    bar = "█" * bar_len
    print(f"{label:<8} | {bar} ({value:.2f}s)")

print("\nVISUAL COMPARISON (Lower is Better):")
max_time = max(t_csv, t_parquet, t_orc)
print_bar("CSV", t_csv, max_time)
print_bar("ORC", t_orc, max_time)
print_bar("PARQUET", t_parquet, max_time)

# --- STEP 5: ANALYSIS ---
print("\n" + "="*60)
print(" WINNER ANALYSIS")
print("="*60)

winner = "PARQUET" if t_parquet < t_orc else "ORC"

print("1. THE BOTTLENECK:")
print("   CSV was slow because Spark had to read EVERY row and column,")
print("   unzip the file, and parse text into numbers.")

print(f"\n2. THE CHAMPION: {winner}")
print("   Both Parquet and ORC use 'Columnar Storage'. They only read the")
print("   'element' and 'value' columns, skipping the rest. This reduces")
print("   I/O by 60-80%.")

print("\n3. WHEN TO USE WHAT?")
print("   [CSV]     -> Data Interchange. Sending data to Excel/Humans.")
print("   [PARQUET] -> General Spark/Python Analytics. Best ecosystem support.")
print("   [ORC]     -> Hive & Presto. Often compresses slightly better than Parquet.")

spark.stop()
