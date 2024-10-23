import pickle 
import wandb
import h5py
import torch
import json
import os

# api = wandb.Api()


# with h5py.File('879ec621a3cdfe08b4a8a598d91d41b91d89c8404ce27dfa8928947d4b710084/rewards.h5', 'r') as f:
#     for key in f.keys():  # Iterate through all datasets (tensors)
#         tensor = torch.tensor(f[key][:])  # Convert back to a PyTorch tensor
#         print(tensor)

# Read history
history = []
# with open(os.path.join(run_dir, "wandb-history.jsonl"), "r") as f:
#     for line in f:
#         history.append(json.loads(line))


# Project is specified by <entity/project-name>
# runs = api.runs("university-alberta/ppo_plots")

# history_list, summary_list, config_list, name_list = [], [], [], []
# for run in runs: 
#     print(run.name)
#     if run.name != "MountainCar-v0__main_ppo_count__1__1729103253":
#         continue

#     summary_list.append(run.summary._json_dict)
#     history = run.history(samples=500000)
#     print(len(history))
#     history_list.append(history)
#     config_list.append(
#         {k: v for k,v in run.config.items()
#           if not k.startswith('_')})
    
#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#     print(run.name)
#     print(config_list)
#     break
    

# data = {
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list,
#     "history": history_list
#     }

# with open('Data/runs_3.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('Data/runs_3.pickle', 'rb') as handle:
#     b = pickle.load(handle)