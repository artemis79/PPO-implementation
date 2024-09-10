import pandas as pd
import wandb

api = wandb.Api()
entity, project = "university-alberta", "ppo-implementation"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
index = 5
episodic_return = []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    print(run.history())
    index += 1
    if index > 6:
        break
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)
# print(summary_list[1])
# print(name_list[1])

# runs_df.to_csv("project.csv")