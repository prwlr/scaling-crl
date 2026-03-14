import pandas as pd
import mlflow
import wandb

# Get the run to export
api = wandb.Api()
run = api.run("/oskar-e-moberg/scaling-crl/runs/szrslpo9")

# Create experiment
experiment_id = mlflow.create_experiment(run.project)

# Log runs under specific experiment
with mlflow.start_run(experiment_id=experiment_id, run_name=run.name):

    # Each row is a step in the original experiment
    for index, row in run.history().iterrows():
        # Log metrics
        for key, value in row.items():
            # Unable to log dicts as metrics in mlflow
            if isinstance(value, dict):
                # TODO: Try to log as artifact?
                continue

            # Log the metric under a key and step
            mlflow.log_metric(key, value, step=index)

        # Log parameters
        # for key, value in row.items():
            # mlflow.log_param(key.strip(), value.strip())

        # Optionally log tags
        # for key, value in row.items():
        #     mlflow.set_tag(key.strip(), value.strip())

