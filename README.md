# Spark Weather Analysis Project

Big Data Analysis on constrained hardware (8GB RAM) using Apache Spark, Python, and Hadoop.

## Project Overview

This project documents the journey of building a Big Data ETL (Extract, Transform, Load) and Analysis pipeline on a single-node Linux laptop with 8GB RAM.

The goal was to move beyond simple CSV processing and utilize Apache Spark's in-memory computation capabilities to ingest, clean, and model meteorological data from the NOAA Global Historical Climatology Network (GHCN). It includes performance benchmarking between CSV, Parquet, and ORC formats to demonstrate the efficiency of columnar storage.

## Repository Structure

### 1. `noaa_weather_analysis.py` (Main Production Script)

The final, optimized pipeline. It performs the following:

- **ETL**: Downloads raw 2024 weather data, cleans it, and pivots it from "Tall" to "Wide" format.
- **Analysis**: Calculates seasonal trends and extreme weather events.
- **Statistics**: Runs Pearson & Spearman dependency tests (Correlation Matrices).
- **Visualization**: Renders Terminal-based Heatmaps and Scatter Plots using plotext.
- **Safety**: Optimized with `.limit()` clauses and memory caps to prevent JVM crashes on 8GB RAM.

### 2. `format_battle.py` (Performance Benchmark)

A script designed to stress-test storage formats. It runs the same aggregation pipeline on three different file types to measure I/O performance.

- **Formats Tested**: CSV (Row-based) vs. Parquet (Columnar) vs. ORC (Optimized Row Columnar).
- **Outcome**: Proved that Columnar formats are 5x-9x faster than CSV for analytical queries.

### 3. `weather_analysis.py` (Initial Prototype)

**Status**: Deprecated / Educational Reference

This was the first attempt at the analysis.

- **Issue Discovered**: The initial sample dataset was corrupted. It expected 2,922 rows but only processed 366 rows.
- **Lesson**: This failure prompted the switch to the robust NOAA GHCN pipeline used in the main script.

### 4. `Spark Weather Analysis Project Complete Journey Report.docx`

A full documentation file detailing the environment setup, error logs, debugging steps (solving Py4JNetworkError), and the theoretical difference between Hadoop MapReduce and Spark DAG execution.

## Key Results & Visuals

### 1. Storage Format Showdown

We ran a benchmark to process ~4.2 million data points. The results highlight why Big Data relies on Columnar storage.

| Format   | Time (Seconds) | Speedup vs CSV | Why? |
|----------|----------------|----------------|------|
| CSV      | 75.80s         | 1.0x (Baseline)| Slow text parsing & row scanning |
| Parquet  | 13.72s         | 5.52x Faster   | Column pruning & efficient compression |
| ORC      | 8.12s          | 9.33x Faster   | Best compression for this specific query type |

### 2. Statistical Correlation (Terminal Output)

We analyzed the relationship between Minimum Temperature (TMIN) and Maximum Temperature (TMAX).

- **Correlation Coefficient**: 0.8993 (Strong Positive Dependency)

*(Add your Terminal Heatmap screenshot here)*

*(Add your Scatter Plot screenshot here)*

## Technology Stack & Environment

- **OS**: Zorin OS (Linux)
- **Hardware**: 8GB RAM, i5 Processor
- **Engine**: Apache Spark 3.5.0 (Local Mode `local[*]`)
- **Storage API**: Hadoop Common Libraries (HDFS Client)
- **Language**: Python 3.10 (PySpark)
- **Libraries**: pyspark, findspark, plotext, numpy

## Deep Dive: Spark DAG vs. Hadoop MapReduce

One of the key learning outcomes of this project was understanding why Spark is faster than traditional Big Data tools. Even though we installed Hadoop, this project does not use the Hadoop MapReduce engine.

**The Old Way (MapReduce)**: Traditional Hadoop jobs write intermediate results to the hard disk after every step (Map → Disk → Reduce → Disk). This causes massive I/O bottlenecks.

**The New Way (Spark DAG)**: Spark builds a Directed Acyclic Graph (DAG) of the entire pipeline before execution.

- **Lazy Evaluation**: It waits until the final `.count()` or `.show()` action to run.
- **Pipelining**: It collapses multiple steps (Filter + Pivot + Aggregate) into a single in-memory pass.
- **In-Memory Speed**: Data stays in RAM (Cache), avoiding the slow hard drive.

**Impact**: This architecture is why our complex correlation analysis ran in seconds on an 8GB laptop, whereas a traditional MapReduce job would have struggled with disk thrashing.

## How to Run

### Prerequisites

Ensure Java 8/11 and Python 3 are installed:

```bash
sudo apt install default-jdk python3-pip
```

### Setup Spark

Download Spark 3.5.0+ and set SPARK_HOME:

```bash
pip install pyspark findspark plotext
```

### Run the Analysis

```bash
python3 noaa_weather_analysis.py
```

### Run the Benchmark

```bash
python3 format_battle.py
```

## Key Learnings

1. **Columnar Formats Superiority**: Parquet and ORC significantly outperform CSV for analytical workloads
2. **Memory Management**: Proper configuration is crucial for running Spark on resource-constrained hardware
3. **Data Validation**: Always verify data integrity before analysis
4. **Spark Architecture**: Understanding DAG execution and lazy evaluation is key to optimization
5. **Error Handling**: Distinguishing between warnings and critical errors in Spark logs

## Future Improvements

1. Implement real-time streaming of weather data
2. Add more sophisticated machine learning models
3. Deploy to a cloud cluster for scalability testing
4. Create interactive dashboards for visualization
5. Implement automated data quality checks
