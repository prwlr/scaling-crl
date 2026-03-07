import pandas as pd
import mlflow

# Load the CSV file
csv_file_path = 'path/to/your/experiments.csv'
experiments_df = pd.read_csv(csv_file_path)

# Start logging experiments
for index, row in experiments_df.iterrows():
    with mlflow.start_run():
        # Log parameters
        for param in row['parameters'].split(';'):
            key, value = param.split('=')
            mlflow.log_param(key.strip(), value.strip())

        # Log metrics
        for metric in row['metrics'].split(';'):
            key, value = metric.split('=')
            mlflow.log_metric(key.strip(), float(value.strip()))

        # Optionally log tags
        for tag in row['tags'].split(';'):
            key, value = tag.split('=')
            mlflow.set_tag(key.strip(), value.strip())

