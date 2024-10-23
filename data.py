import pickle 
import wandb
import h5py
import torch
import json
import os

# api = wandb.Api()




# Read history
history = []


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