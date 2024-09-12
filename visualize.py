import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np


api = wandb.Api()
entity, project = "university-alberta", "ppo-implementation"
runs = api.runs(entity + "/" + project)


episodic_return = []
episode_number = []
min_episode_number = float('inf')

for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    df = run.history(samples=5000, x_axis="episode_number", keys=["episodic_return_per_episode"])
    if "episodic_return_per_episode" in df and "episode_number" in df:
        # print(df)
        episodic_return.append(np.array(df["episodic_return_per_episode"]))
        episode_number = df["episode_number"].tolist()
        min_episode_number = min(min_episode_number, episode_number[-1])

returns = []
for ele in episodic_return:
    returns.append(ele[:][0 : min_episode_number])


average_returns = np.mean(returns, axis=0)
std_returns = np.std(returns, axis=0)
x = episode_number[0: int(min_episode_number)]
confidence_interval = 1.96* std_returns/np.sqrt(len(returns))
plt.plot(x, average_returns)
plt.fill_between(x, average_returns-confidence_interval, average_returns+confidence_interval, alpha=0.5)
plt.show()
