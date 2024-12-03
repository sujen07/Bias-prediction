import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_csv():
    data_dir = "data"

    final_labeled_data = [file for file in os.listdir(data_dir) if file.endswith('.csv') and file.startswith('final_')]

    dataframes = []

    # Loop through the filtered list of CSV files and read each one
    for file in final_labeled_data:
        print(file)
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, delimiter=';')
        df = df[['text', 'type', 'label_bias', 'biased_words']]
        df = df[df['label_bias'] != 'No agreement']
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df['label_bias'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_bias'])
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    train_df, val_df, test_df = preprocess_csv()

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

