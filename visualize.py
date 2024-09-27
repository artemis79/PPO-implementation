import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid



def parse_args_visualize():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="main_ppo",
        help="the name of the method you want to visualize")
    
    parser.add_argument("--aggregate-function", type=str, default="mean",
        help="")

    parser.add_argument("--load", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    api = wandb.Api()
    entity, project = "university-alberta", "ppo"
    runs = api.runs(entity + "/" + project)


    episodic_return = []
    episode_number = []
    min_episode_number = float('inf')
    args = parse_args_visualize()
    
    if args.load:
        x_positions_ppo = np.load('tmp/x_positions__' + 'main_ppo' + '.npy')
        x_positions_count  = np.load('tmp/x_positions__'  + 'main_ppo_count' + '.npy')
        velocities_ppo = np.load('tmp/velocities__'  + 'main_ppo' + '.npy')
        velocities_count = np.load('tmp/velocities__'  + 'main_ppo_count' + '.npy')
        algorithms = {"PPO": [x_positions_ppo, velocities_ppo], "PPO with count": [x_positions_count, velocities_count]}
        print([v for v in velocities_count if v > 0.07])

    else:
        x_positions = []
        velocities = []
        for run in runs:
            exp_name = run.config['exp_name']
            gym_id = run.config['gym_id']
            agg_func = run.config['aggregate_function']
            print(exp_name, run.config['seed'])
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            df = run.history(samples=500000)
            i = 0
            if exp_name == args.exp_name and gym_id == "MountainCar-v0" and  "observation" and args.aggregate_function == agg_func:
                positions = df["observation"].to_numpy()
                for position in positions:
                    if position and type(position) != float: 
                        print(position)
                        for x, v in position:
                            x_positions.append(x)
                            velocities.append(v)

                
        
        np.save('tmp/x_positions__' + args.exp_name + "__" + args.aggregate_function, x_positions)
        np.save('tmp/velocities__'  + args.exp_name + "__" + args.aggregate_function, velocities )

    if args.load:
       
        # plt.bars(heatmap, bins)

        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 1),
                axes_pad=0.5,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                )   
        i = 0

        for algo in algorithms.keys():
            x = algorithms[algo][0]
            ax = grid[i]
            extent = [-1.2, 0.6, 0, 1]
            bins = np.linspace(extent[0], extent[1], num=50)
            heatmap, xedges = np.histogram(x, bins=bins)
            cax = ax.imshow(heatmap[np.newaxis,:], cmap="hot", aspect="auto", extent=extent)

            ax.set_yticks([])
            ax.set_xlim(extent[0], extent[1])
            ax.set_title(algo)
            i += 1

        grid.cbar_axes[0].colorbar(cax)

        for cax in grid.cbar_axes:
            cax.toggle_label(True)



        # plt.pcolormesh(heatmap, cmap='Greys', shading='gouraud')
        plt.savefig('figures/MountainCar-v0-heatmap')
        plt.close()




        fig = plt.figure(figsize=(8, 10))
        grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 1),
                axes_pad=0.5,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                )   
        i = 0

        for algo in algorithms.keys():
            x = algorithms[algo][0]
            v = algorithms[algo][1]
            ax = grid[i]
            extent = [-1.2, 0.6, -0.07, 0.07]
            heatmap, xedges, vedges = np.histogram2d(x, v, bins=50)
            cax = ax.imshow(heatmap.T, cmap="hot", aspect=10, extent=extent)

            ax.set_yticks([-0.07, 0.07])
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(algo)
            i += 1

        grid.cbar_axes[0].colorbar(cax)

        for cax in grid.cbar_axes:
            cax.toggle_label(True)

        plt.savefig('figures/MountainCar-v0-position-velocity')
        plt.close()

        

