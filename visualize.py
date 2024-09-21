import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import argparse


def parse_args_visualize():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="main_ppo",
        help="the name of the method you want to visualize")

    parser.add_argument("--load", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    api = wandb.Api()
    entity, project = "university-alberta", "ppo-occupancy"
    runs = api.runs(entity + "/" + project)


    episodic_return = []
    episode_number = []
    min_episode_number = float('inf')
    args = parse_args_visualize()
    
    if args.load:
        x_positions = np.load('tmp/x_positions__' + args.exp_name + '.npy')
        velocities  = np.load('tmp/velocities__'  + args.exp_name + '.npy')

    else:
        x_positions = []
        velocities = []
        for run in runs:
            exp_name = run.config['exp_name']
            gym_id = run.config['gym_id']
            print(exp_name, run.config['seed'])
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            df = run.history(samples=50000)
            i = 0

            if exp_name == args.exp_name and gym_id == "MountainCar-v0" and  "observation" in df:
                positions = df["observation"].to_numpy()
                for position in positions:
                    if position and type(position) != float: 
                        print(position)
                        for x, v in position:
                            x_positions.append(x)
                            velocities.append(v)

                
        
        np.save('tmp/x_positions__' + args.exp_name, x_positions)
        np.save('tmp/velocities__'  + args.exp_name, velocities )



    print(x_positions[0: 50])
    heatmap, xedges, vedges = np.histogram2d(x_positions, velocities, bins=50)
    extent = [-1.5, 0.5, -1, 1]

    fig, ax = plt.subplots()
    occupancy = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Blues')
    fig.colorbar(occupancy, ax=ax)

    plt.show()
