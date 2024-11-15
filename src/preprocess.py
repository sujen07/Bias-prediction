import os
import pandas as pd

data_dir = "data"

final_labeled_data = [file for file in os.listdir(data_dir) if file.endswith('.csv') and file.startswith('final_')]

dataframes = []

# Loop through the filtered list of CSV files and read each one
for file in final_labeled_data:
    print(file)
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path, delimiter=';')
    df = df[['text', 'type', 'label_bias']]
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
