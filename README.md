# HB Meter - Enhanced PPG Feature Engineering

This project provides a complete pipeline for extracting and engineering advanced features from Photoplethysmography (PPG) signals to support hemoglobin prediction research.

The pipeline processes videos recorded under Red, Orange, and Yellow light. For each video, it produces an aggregated summary of all detected heartbeats, including standard statistics, Heart Rate Variability (HRV) metrics, and Pulse Wave Morphology features.

## Enhanced Features

In addition to basic statistics, this pipeline engineers the following high-value features for each video:

-   **Heart Rate Variability (HRV):**
    -   `avg_heart_rate_bpm`: Average heart rate.
    -   `sd_ibi_ms`: Standard Deviation of Inter-Beat Intervals (SDNN), a key HRV metric.
    -   `rmssd_ms`: Root Mean Square of Successive Differences, a measure of short-term variability.
-   **Pulse Wave Morphology (Shape):**
    -   `avg_rise_time_s`: Average time from diastolic trough to systolic peak.
    -   `skew_pulse_amplitude`: Skewness of the beat amplitude distribution.
    -   `kurt_pulse_amplitude`: Kurtosis (tailedness) of the beat amplitude distribution.

## Project Structure & Output

The final output is three distinct CSV files (`RED_results.csv`, `ORANGE_results.csv`, `YELLOW_results.csv`), each containing a rich feature set for every patient video, ready for machine learning.

## Setup

1.  **Navigate to the project directory:** `cd sd_analysis`
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install requirements:** `pip install -r requirements.txt`
4.  **Verify Data Paths:** **Crucially, check the `DATASET_DIR` and `DATA_CSV_PATH` at the top of `main.py` and ensure they are correct for your system.**

## Usage

1.  **Run the feature engineering pipeline:** `python main.py`
2.  **Run the correlation analysis:** `python correlation_analysis.py` (This will now analyze all the new features).
