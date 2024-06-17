import os
import pandas as pd
import pathlib
metrics_pretty_names = {
    'accuracy': 'Accuracy',
    'average_total_throughput': 'Average Total Throughput',
    'standard_deviation_total_throughput': 'Standard Deviation Total Throughput',
    'tpr': 'TPR',
    'fpr': 'FPR',
    'average_throughput': 'Average Throughput',
    'standard_deviation_throughput': 'Standard Deviation Throughput',
    'model': 'Model'
    
}
pathlib.Path('views/').mkdir(parents=True, exist_ok=True) 
# Load the data
df = pd.read_json('data/results.json', lines=True)
df_fids = pd.read_json('data/results_fids.json', lines=True)
df = pd.concat([df, df_fids], axis=0)
# Ensure the output_date is in datetime format
df['output_date'] = pd.to_datetime(df['output_date'])

# Sort the DataFrame by output_date to get the latest entries for each model
df_sorted = df.sort_values(by='output_date', ascending=False)

# Filter to get the latest result for each model and number of packets combination
df_latest = df_sorted.drop_duplicates(subset=['model', 'number_packets'], keep='first')

# Separate the results_fids models from the others
df_fids_models = df_latest[df_latest['model'].isin(df_fids['model'])]
df_other_models = df_latest[~df_latest['model'].isin(df_fids['model'])]

# Concatenate the fids models at the top
df_latest = pd.concat([df_fids_models, df_other_models], axis=0)
import constants
# Define the function to generate LaTeX table and save to a file
def generate_latex_table(df, metric, file_path):
    # Pivot the DataFrame to have model as the index and number_packets as columns
    # print(df)
    npoptions=constants.NUMBER_PACKETS_OPTIONS + ['full']
    df_pivot = df.pivot(index='model', columns='number_packets', values=metric).reindex(index=df['model'].unique(), columns=npoptions)
    df_pivot.reset_index(inplace=True)
    # Define the columns for the table
    columns = ['model'] + [col for col in df_pivot.columns if col not in ['model', 'confusion_matrix']]
    columns_packet_numbers = df_pivot.columns[1:]
    columns_packet_numbers = list(columns_packet_numbers)
    columns_packet_numbers.remove('full')
    
    # Extract the packet numbers for the caption
    packet_numbers = ", ".join(map(str, columns_packet_numbers))  # Exclude 'Model' column
    
    # LaTeX table generation function with multi-row header
    def df_to_latex_pivot(df, columns):
        columns_str = list(map(str, columns))
        # Number of packets columns
        num_packet_cols = columns_str[1:]
        
        latex_table = "\\begin{table}[H]\n\\centering\n"+r'\begin{adjustbox}{width=\textwidth}'+"\n\\begin{tabular}{|c|" + "c|" * len(num_packet_cols) + "}\n\\hline\n"
        
        # Multi-row header
        latex_table += "\\multirow{2}{*}{Model} & \\multicolumn{" + str(len(num_packet_cols)) + "}{c|}{Number of Packets} \\\\\n\\cline{2-" + str(len(columns)) + "}\n"
        latex_table += " & " + " & ".join(num_packet_cols) + " \\\\\n\\hline\n"
        
        for _, row in df.iterrows():
            latex_table += " & ".join([f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in columns]) + " \\\\\n"
        
        latex_table += "\\hline\n\\end{tabular}\n"+r'\end{adjustbox}'+"\n\\caption{" + metrics_pretty_names[metric] + " by Model for Flows with at Most " + packet_numbers + " Packets, and Complete Flows}\n\\label{tab:" + metric.lower().replace(" ", "_") + "_results}\n\\end{table}"
        return latex_table

    # Generate the LaTeX table
    latex_code = df_to_latex_pivot(df_pivot, columns)
    
    # Save the LaTeX table to a file
    with open(file_path, 'w') as f:
        f.write(latex_code)

# Identify all metric columns (excluding non-metric columns)
non_metric_columns = ['model', 'number_packets', 'output_date', 'best_parameters', 'timeout']
metric_columns = ['accuracy', 'average_total_throughput', 'standard_deviation_total_throughput']

# Generate LaTeX tables for all metrics and save to files
for metric in metric_columns:
    file_path = f'views/{metric.lower().replace(" ", "_").replace("/", "_")}_table.tex'
    generate_latex_table(df_latest, metric, file_path)

# Load the data
# df = pd.read_json('data/results.json', lines=True)
# df_fids = pd.read_json('data/results_fids.json', lines=True)
# df = pd.concat([df, df_fids], axis=0)

# # Ensure the output_date is in datetime format
# df['output_date'] = pd.to_datetime(df['output_date'])

# Filter the DataFrame to include only results for 'full' number of packets
df_full_packets = df_latest[df_latest['number_packets'] == 'full']

# # Sort the DataFrame by output_date to get the latest entries for each model
# df_sorted = df_full_packets.sort_values(by='output_date', ascending=False)

# # Filter to get the latest result for each model and number of packets combination
# df_latest_full = df_sorted.drop_duplicates(subset=['model'], keep='first')

# # Separate the results_fids models from the others
# df_fids_full_models = df_latest_full[df_latest_full['model'].isin(df_fids['model'])]
# df_other_full_models = df_latest_full[~df_latest_full['model'].isin(df_fids['model'])]

# # Concatenate the fids models at the top
# df_latest_full = pd.concat([df_fids_full_models, df_other_full_models], axis=0)

# Define the function to generate LaTeX table and save to a file
def generate_latex_table_full(df, file_path):
    # Define the columns for the table
    # excluded_columns = [
    #     'model', 'number_packets', 'output_date', 'best_parameters', 'timeout', 
    #     'confusion_matrix', '', 'precision', 'recall', 'confusion_matrix', 'Average Prediction Time (s/sample)', 
    #     'Standard Deviation Prediction Time (s/sample)', 
    #     'Average Throughput (samples/s)', 
    #     'Standard Deviation Throughput (samples/s)'
    # ]

    # columns = ['model'] + [col for col in df.columns if col not in excluded_columns]
    columns= ['model','accuracy','tpr','fpr',"average_throughput","standard_deviation_throughput","average_total_throughput","standard_deviation_total_throughput"]
    # LaTeX table generation function
    def df_to_latex_pivot(df, columns):
        columns_str = list(map(str, columns))
        latex_table = "\\begin{table}[H]\n\\centering\n"+r'\begin{adjustbox}{width=\textwidth}'+"\n\\begin{tabular}{|" + " | ".join(["c"] * len(columns)) + "|}\n\\hline\n"
        latex_table += " & ".join(map(lambda x : metrics_pretty_names[x], columns_str)) + " \\\\\n\\hline\n"
        for _, row in df.iterrows():
            latex_table += " & ".join([f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in columns]) + " \\\\\n"

        latex_table += "\\hline\n\\end{tabular}"+r"\end{adjustbox}"+"\n\\caption{Results across multiple metrics with all complete flows.}\n\\label{tab:full_packets_results}\n\\end{table}"
        return latex_table

    # Generate the LaTeX table
    latex_code = df_to_latex_pivot(df, columns)
    
    # Save the LaTeX table to a file
    with open(file_path, 'w') as f:
        f.write(latex_code)

# Define the file path for the LaTeX table
file_path = 'views/full_packets_results.tex'

# Generate LaTeX table for full number of packets and save to a file
generate_latex_table_full(df_full_packets, file_path)
