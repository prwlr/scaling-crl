import pandas as pd
import mlflow
import wandb
import os

# Get the run to export
api = wandb.Api()
run: wandb.Run = api.run("/oskar-e-moberg/scaling-crl/runs/szrslpo9")

# Create experiment
experiment = mlflow.get_experiment_by_name(run.project)
if experiment is None:
    experiment_id = mlflow.create_experiment(run.project)
else:
    experiment_id = experiment.experiment_id

# Log runs under specific experiment
with mlflow.start_run(experiment_id=experiment_id, run_name=run.name):

    # Each row is a step in the original experiment
    for index, row in run.history().iterrows():
        # Log metrics
        for key, value in row.items():

            # Log the html-visualisation as an artifact
            if isinstance(value, dict) and key == "vis":
                for path in os.listdir("./wandb"):
                    if run.path[-1] in path:
                        mlflow.log_artifact("./wandb/" + path + "/files/" + value["path"])
                        break
                continue

            # Log the metric under a key and step
            mlflow.log_metric(key, value, step=index)

            # Log parameters
            # for key, value in row.items():
                # mlflow.log_param(key.strip(), value.strip())

            # Optionally log tags
            # for key, value in row.items():
            #     mlflow.set_tag(key.strip(), value.strip())

