# ==============================================================================
#           HB METER ANALYSIS - SEGMENTED VIDEO (FINAL VERSION)
# ==============================================================================
# This script processes a single 30-second video per patient, slicing it
# into RED, ORANGE, and YELLOW segments. It produces:
# 1. Aggregated summary CSVs for each color with detailed peak statistics.
# 2. A detailed CSV with every single detected heartbeat's delta value AND
#    all corresponding patient metadata (including hb).
# ==============================================================================

import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import tempfile
import contextlib
import cv2
from scipy.signal import butter, filtfilt, detrend, find_peaks
import matplotlib.pyplot as plt

# --- Configuration: Use RELATIVE paths for portability ---
DATASET_DIR = '../hb_meter_project_code/HB_new_dataset'
DATA_CSV_PATH = '../hb_meter_project_code/hb_dataset_new.csv' # Using your ORIGINAL csv
# -----------------------------------------------------------

LOG_FILE = 'analysis.log'
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
PER_BEAT_CSV_PATH = os.path.join(OUTPUT_DIR, 'per_beat_delta_values.csv') # Path for the detailed CSV


# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

@contextlib.contextmanager
def safe_video_path(path_with_spaces):
    """Creates a temporary, space-free copy of a video for safe processing."""
    safe_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            safe_path = tmp.name
        shutil.copy2(path_with_spaces, safe_path)
        yield safe_path
    finally:
        if safe_path and os.path.exists(safe_path):
            os.remove(safe_path)

def setup_directories(dir_list):
    """Creates directories if they don't already exist."""
    for directory in dir_list:
        try: os.makedirs(directory, exist_ok=True)
        except OSError as e: logging.error(f"Error creating directory {directory}: {e}"); raise

def load_and_prepare_data(csv_path, dataset_dir):
    """Loads data from the CSV and prepares it for analysis."""
    if not os.path.exists(csv_path): logging.error(f"CSV not found: {csv_path}"); return None
    try: df = pd.read_csv(csv_path)
    except Exception as e: logging.error(f"Failed to read CSV: {e}"); return None
    if 'name' not in df.columns and 'patient_id' not in df.columns: logging.error("Col 'name'/'patient_id' not found"); return None
    if 'name' in df.columns: df.rename(columns={'name': 'patient_id'}, inplace=True)
    if 'matched_video' not in df.columns: logging.error("Col 'matched_video' not found"); return None
    df.dropna(subset=['matched_video'], inplace=True)
    df['full_video_path'] = df['matched_video'].apply(lambda f: os.path.join(dataset_dir, str(f)))
    df = df[df['full_video_path'].apply(os.path.exists)].copy()
    logging.info(f"Prepared data for {len(df)} videos.")
    return df

def extract_ppg_signal(video_path):
    """Extracts a raw PPG signal from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.warning(f"OpenCV could not open: {video_path}"); return None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: logging.warning(f"Video has zero FPS: {video_path}"); return None, None
        raw_signal = [np.mean(frame[:, :, 2]) for ret, frame in iter(lambda: cap.read(), (False, None))]
        cap.release()
        if not raw_signal: logging.warning(f"No frames read from: {video_path}"); return None, None
        return np.array(raw_signal), fps
    except Exception as e:
        logging.error(f"Error during signal extraction for {video_path}: {e}"); return None, None

def process_ppg_signal(raw_signal, fps):
    """Applies filtering to the raw PPG signal."""
    detrended = detrend(-raw_signal)
    nyquist = 0.5 * fps
    low, high = 0.7/nyquist, 4.0/nyquist
    if high >= 1: high = 0.99
    if low >= high: logging.warning("Cannot create filter, low_cut > high_cut"); return None
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, detrended)

def find_ppg_peaks(filtered_signal, fps):
    """Finds systolic and diastolic peaks."""
    min_dist = int(fps * 60 / 220)
    prominence = np.std(filtered_signal) * 0.3
    s_peaks, _ = find_peaks(filtered_signal, distance=min_dist, prominence=prominence)
    d_troughs, _ = find_peaks(-filtered_signal, distance=min_dist, prominence=prominence)
    return s_peaks, d_troughs

def calculate_per_beat_metrics(systolic_peaks, diastolic_troughs, signal, fps):
    """Calculates detailed metrics for each detected heartbeat."""
    metrics = []
    if len(systolic_peaks) < 2 or len(diastolic_troughs) == 0: return metrics
    for i in range(len(systolic_peaks)):
        peak_idx = systolic_peaks[i]
        preceding_troughs = diastolic_troughs[diastolic_troughs < peak_idx]
        if len(preceding_troughs) == 0: continue
        trough_idx = preceding_troughs[-1]
        
        systole_val = signal[peak_idx]
        diastole_val = signal[trough_idx]
        ibi_s = (systolic_peaks[i] - systolic_peaks[i-1]) / fps if i > 0 else np.nan
        
        metrics.append({
            'systole_value': systole_val,
            'diastole_value': diastole_val,
            'pulse_amplitude': systole_val - diastole_val,
            'delta': systole_val + diastole_val,
            'inter_beat_interval_s': ibi_s,
            'rise_time_s': (peak_idx - trough_idx) / fps
        })
    return metrics

def plot_ppg_analysis(signal, fps, s_peaks, d_troughs, p_id, color, out_dir):
    """Generates and saves a plot of the PPG analysis for a segment."""
    time = np.arange(len(signal)) / fps
    plt.figure(figsize=(15, 6))
    plt.plot(time, signal, label=f'Filtered PPG ({color})', color='royalblue', lw=1.5)
    plt.scatter(time[s_peaks], signal[s_peaks], c='red', s=80, marker='^', label='Systolic', zorder=5)
    plt.scatter(time[d_troughs], signal[d_troughs], c='limegreen', s=80, marker='v', label='Diastolic', zorder=5)
    safe_id = str(p_id).replace('/', '_')
    plt.title(f'PPG Analysis for {safe_id} - {color} Segment', fontsize=16)
    plt.xlabel('Time in Segment (s)'); plt.ylabel('Amplitude (A.U.)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(out_dir, f"{safe_id}_{color}_segment_analysis.png"))
    plt.close()


# ==============================================================================
#                          MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
    
    logging.info("--- Starting Segmented Video Analysis ---")
    setup_directories([OUTPUT_DIR, PLOTS_DIR])

    patient_df = load_and_prepare_data(DATA_CSV_PATH, DATASET_DIR)
    if patient_df is None or patient_df.empty: logging.critical("No data. Exiting."); return

    all_summaries, all_beat_data = [], []
    success_count = 0
    failure_reasons = {'extraction': 0, 'processing': 0, 'no_beats': 0, 'other': 0}

    for index, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="Analyzing Videos"):
        original_path = row['full_video_path']
        try:
            with safe_video_path(original_path) as path_to_process:
                full_signal, fps = extract_ppg_signal(path_to_process)
                if full_signal is None or fps is None or fps == 0:
                    failure_reasons['extraction'] += 1
                    continue
            
            ten_seconds_in_frames = int(10 * fps)
            segments = {
                'RED':    full_signal[0 : ten_seconds_in_frames],
                'ORANGE': full_signal[ten_seconds_in_frames : 2 * ten_seconds_in_frames],
                'YELLOW': full_signal[2 * ten_seconds_in_frames : ]
            }

            video_had_success = False
            for color, signal_segment in segments.items():
                if len(signal_segment) < fps * 2: continue
                    
                filtered_signal = process_ppg_signal(signal_segment, fps)
                if filtered_signal is None: failure_reasons['processing'] += 1; continue

                s_peaks, d_troughs = find_ppg_peaks(filtered_signal, fps)
                beat_metrics = calculate_per_beat_metrics(s_peaks, d_troughs, filtered_signal, fps)
                if len(beat_metrics) < 3: failure_reasons['no_beats'] += 1; continue
                
                video_had_success = True
                plot_ppg_analysis(filtered_signal, fps, s_peaks, d_troughs, row['patient_id'], color, PLOTS_DIR)
                
                # --- MODIFIED: Add all patient metadata to each beat ---
                for i, beat in enumerate(beat_metrics):
                    beat_info = {
                        'light_color': color,
                        'beat_number': i + 1,
                        'delta': beat['delta'],
                        'pulse_amplitude': beat['pulse_amplitude']
                    }
                    # Merge the full patient row with the specific beat's data
                    all_beat_data.append({**row.to_dict(), **beat_info})

                beats_df = pd.DataFrame(beat_metrics).dropna()
                if len(beats_df) < 3: continue
                
                ibi_s = beats_df['inter_beat_interval_s']
                summary = {
                    'systole_min': beats_df['systole_value'].min(), 'systole_max': beats_df['systole_value'].max(),
                    'systole_mean': beats_df['systole_value'].mean(), 'systole_median': beats_df['systole_value'].median(), 'systole_sd': beats_df['systole_value'].std(ddof=0),
                    'diastole_min': beats_df['diastole_value'].min(), 'diastole_max': beats_df['diastole_value'].max(),
                    'diastole_mean': beats_df['diastole_value'].mean(), 'diastole_median': beats_df['diastole_value'].median(), 'diastole_sd': beats_df['diastole_value'].std(ddof=0),
                    'num_systole_peaks_raw': len(s_peaks), 'num_diastole_peaks_raw': len(d_troughs), 'num_valid_beats': len(beats_df),
                    'avg_heart_rate_bpm': 60/ibi_s.mean(), 'sd_ibi_ms': ibi_s.std(ddof=0)*1000,
                    'rmssd_ms': np.sqrt(np.mean(np.diff(ibi_s)**2))*1000, 'avg_rise_time_s': beats_df['rise_time_s'].mean(),
                    'avg_pulse_amplitude': beats_df['pulse_amplitude'].mean(), 'sd_pulse_amplitude': beats_df['pulse_amplitude'].std(ddof=0),
                    'skew_pulse_amplitude': beats_df['pulse_amplitude'].skew(), 'kurt_pulse_amplitude': beats_df['pulse_amplitude'].kurt(),
                    'avg_delta': beats_df['delta'].mean(), 'sd_delta': beats_df['delta'].std(ddof=0),
                }
                all_summaries.append({**row.to_dict(), **summary, 'light_color': color})
            
            if video_had_success: success_count += 1

        except Exception as e:
            logging.error(f"CRITICAL FAILURE on {original_path}: {e}")
            failure_reasons['other'] += 1
            continue

    # --- FINAL REPORT AND SAVING ---
    logging.info("\n" + "="*50 + "\n--- FINAL ANALYSIS REPORT ---\n" + "="*50)
    # ... (Reporting logic remains the same)

    if all_summaries:
        master_df = pd.DataFrame(all_summaries)
        for color in ['RED', 'ORANGE', 'YELLOW']:
            color_df = master_df[master_df['light_color'] == color].copy()
            color_df.drop(columns=['light_color', 'matched_video', 'full_video_path'], inplace=True, errors='ignore')
            if not color_df.empty:
                output_path = os.path.join(OUTPUT_DIR, f'{color}_results.csv')
                color_df.to_csv(output_path, index=False, float_format='%.4f')
                logging.info(f"Saved {len(color_df)} aggregated records to {output_path}")

    # --- MODIFIED: Save the PER-BEAT file with all metadata ---
    if all_beat_data:
        logging.info("\n--- Saving Per-Beat Detailed Data ---")
        per_beat_df = pd.DataFrame(all_beat_data)
        # Drop columns we don't need in this detailed view
        per_beat_df.drop(columns=['matched_video', 'full_video_path'], inplace=True, errors='ignore')
        per_beat_df.to_csv(PER_BEAT_CSV_PATH, index=False, float_format='%.4f')
        logging.info(f"Saved {len(per_beat_df)} individual beat records to {PER_BEAT_CSV_PATH}")
    else:
        logging.warning("No per-beat data was generated to save.")

    logging.info("\n--- Analysis Finished ---\n" + "="*50)

if __name__ == "__main__":
    main()