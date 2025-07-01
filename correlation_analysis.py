import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

OUTPUT_DIR = 'output'
LOG_FILE = 'correlation.log'

def analyze_correlations(color):
    results_file = os.path.join(OUTPUT_DIR, f'{color}_results.csv')
    if not os.path.exists(results_file): return

    df = pd.read_csv(results_file)
    logging.info(f"--- Analyzing {color} Data ({len(df)} records) ---")
    
    features = [
        'avg_heart_rate_bpm', 'sd_ibi_ms', 'rmssd_ms',
        'avg_rise_time_s', 'avg_pulse_amplitude', 'sd_pulse_amplitude',
        'skew_pulse_amplitude', 'kurt_pulse_amplitude', 'avg_delta'
    ]
    
    df['hb'] = pd.to_numeric(df['hb'], errors='coerce')
    for feat in features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
    df.dropna(subset=['hb'] + features, inplace=True)
    
    if len(df) < 3:
        logging.warning(f"Not enough clean data for {color}."); return

    correlation_matrix = df[['hb'] + features].corr()
    hb_correlations = correlation_matrix['hb'].drop('hb').sort_values(ascending=False)
    logging.info(f"Correlation with Hemoglobin (hb) for {color}:\n{hb_correlations}\n")
    
    # Visualize the top 2 positively and negatively correlated features
    top_pos = hb_correlations.head(2).index.tolist()
    top_neg = hb_correlations.tail(2).index.tolist()
    top_features = top_pos + top_neg
    
    if not top_features: return
    
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 2, i + 1)
        sns.regplot(data=df, x='hb', y=feature, line_kws={"color": "red"})
        plt.title(f'Hb vs. {feature.replace("_", " ").title()}')
        plt.grid(True, linestyle='--')
    
    plt.suptitle(f'Top Correlated Features for {color} Light (n={len(df)})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(OUTPUT_DIR, f'{color}_top_correlations_plot.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved top correlations plot to {plot_path}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
    for color in ['RED', 'ORANGE', 'YELLOW']:
        analyze_correlations(color)

if __name__ == "__main__":
    main()
