import os
import pandas as pd

# Load the data
df = pd.read_json('data/results.json', lines=True)
df_fids = pd.read_json('data/results_fids.json', lines=True)
df = pd.concat([df,df_fids],axis=0)
# Ensure the output_date is in datetime format
df['output_date'] = pd.to_datetime(df['output_date'])

# Sort the DataFrame by output_date to get the latest entries for each model
df_sorted = df.sort_values(by='output_date', ascending=False)

# Filter to get the latest result for each model and number of packets combination
df_latest = df_sorted.drop_duplicates(subset=['Model', 'number_packets'], keep='first')
print(df_latest)
# Define the function to generate LaTeX table and save to a file
def generate_latex_table(df, metric, file_path):
    # Pivot the DataFrame to have Model as the index and number_packets as columns
    df_pivot = df.pivot(index='Model', columns='number_packets', values=metric)
    
    # Reset the index to ensure Model is a regular column and not an index
    df_pivot.reset_index(inplace=True)

    # Define the columns for the table
    columns = ['Model'] + [col for col in df_pivot.columns if col != 'Model']

    # LaTeX table generation function
    def df_to_latex_pivot(df, columns):
        columns_str = list(map(str, columns))
        latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|" + " | ".join(["c"] * len(columns)) + "|}\n\\hline\n"
        latex_table += " & ".join(columns_str) + " \\\\\n\\hline\n"
        for _, row in df.iterrows():
            latex_table += " & ".join([str(row[col]) for col in columns]) + " \\\\\n"
        latex_table += "\\hline\n\\end{tabular}\n\\caption{" + metric + " by Model and Number of Packets}\n\\label{tab:" + metric.lower().replace(" ", "_") + "_results}\n\\end{table}"
        return latex_table

    # Generate the LaTeX table
    latex_code = df_to_latex_pivot(df_pivot, columns)
    
    # Save the LaTeX table to a file
    with open(file_path, 'w') as f:
        f.write(latex_code)

# Identify all metric columns (excluding non-metric columns)
non_metric_columns = ['Model', 'number_packets', 'output_date', 'Best Parameters', 'timeout']
metric_columns = [col for col in df.columns if col not in non_metric_columns]

# Generate LaTeX tables for all metrics and save to files
for metric in metric_columns:
    file_path = f'views/{metric.lower().replace(" ", "_").replace("/", "_")}_table.tex'
    generate_latex_table(df_latest, metric, file_path)
    
    
    
    
    
    

# Load the data
df = pd.read_json('data/results.json', lines=True)
df_fids = pd.read_json('data/results_fids.json', lines=True)
df = pd.concat([df, df_fids], axis=0)

print(df_latest)


# Ensure the output_date is in datetime format
df['output_date'] = pd.to_datetime(df['output_date'])

# Filter the DataFrame to include only results for 'full' number of packets
df_full_packets = df[df['number_packets'] == 'full']

# Sort the DataFrame by output_date to get the latest entries for each model
df_sorted = df_full_packets.sort_values(by='output_date', ascending=False)

# Filter to get the latest result for each model and number of packets combination
df_latest = df_sorted.drop_duplicates(subset=['Model'], keep='first')
# print(df_latest)
# Define the function to generate LaTeX table and save to a file
def generate_latex_table(df, file_path):
    # Define the columns for the table
    columns = ['Model'] + [col for col in df.columns if col not in ['Model', 'number_packets', 'output_date', 'Best Parameters', 'timeout']]

    # LaTeX table generation function
    def df_to_latex_pivot(df, columns):
        columns_str = list(map(str, columns))
        latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|" + " | ".join(["c"] * len(columns)) + "|}\n\\hline\n"
        latex_table += " & ".join(columns_str) + " \\\\\n\\hline\n"
        for _, row in df.iterrows():
            latex_table += " & ".join([str(row[col]) for col in columns]) + " \\\\\n"
        latex_table += "\\hline\n\\end{tabular}\n\\caption{Results for Full Number of Packets}\n\\label{tab:full_packets_results}\n\\end{table}"
        return latex_table

    # Generate the LaTeX table
    latex_code = df_to_latex_pivot(df, columns)
    
    # Save the LaTeX table to a file
    with open(file_path, 'w') as f:
        f.write(latex_code)

# Define the file path for the LaTeX table
file_path = 'views/full_packets_results.tex'

# Generate LaTeX table for full number of packets and save to a file
generate_latex_table(df_latest, file_path)


