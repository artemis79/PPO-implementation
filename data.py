import pickle 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("university-alberta/ppo-tmp")

history_list, summary_list, config_list, name_list = [], [], [], []
for run in runs: 
    if run.name != "MountainCar-v0__main_ppo_count__1__1727993481":
        continue

    summary_list.append(run.summary._json_dict)
    history = run.history(samples=500000)
    print(len(history))
    history_list.append(history)
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})
    
    # .name is the human-readable name of the run.
    name_list.append(run.name)
    print(run.name)
    break
    

data = {
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
    "history": history_list
    }

with open('Data/runs.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Data/runs.pickle', 'rb') as handle:
    b = pickle.load(handle)