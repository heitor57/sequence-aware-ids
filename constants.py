# NUMBER_PACKETS_OPTIONS = [2,6,20,50]
NUMBER_PACKETS_OPTIONS = []
TIMEOUT = 30


import pandas as pd

def metrics_log_formatter(name, best_params, number_packets, TIMEOUT, accuracy, precision, recall, cm, tpr_micro, fpr_micro,
                      avg_pred_time, std_pred_time, avg_throughput,
                      std_throughput, avg_total_pred_time, std_total_pred_time,
                      avg_total_throughput, std_total_throughput):
    return {
        "model": name,
        "best_parameters": best_params,
        "timeout": TIMEOUT,
        "number_packets": number_packets,
        "output_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "tpr": tpr_micro,
        "fpr": fpr_micro,
        "average_prediction_time": avg_pred_time,
        "standard_deviation_prediction_time": std_pred_time,
        "average_throughput": avg_throughput,
        "standard_deviation_throughput": std_throughput,
        "average_total_prediction_time": avg_total_pred_time,
        "standard_deviation_total_prediction_time": std_total_pred_time,
        "average_total_throughput": avg_total_throughput,
        "standard_deviation_total_throughput": std_total_throughput,
    }
