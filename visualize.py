import wandb
import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import AxesGrid

from ast import literal_eval



def parse_args_visualize():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="main_ppo",
        help="the name of the method you want to visualize")
    parser.add_argument("--load", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    
    args = parser.parse_args()
    return args


def r_intrinsic_plot(r_intrinsic, x_position, velocity, update_number):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = x_position
    Y = velocity
    X, Y = np.meshgrid(X, Y)
    Z = r_intrinsic
    ax.set_title("Update_number:" + str(update_number))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    

if __name__ == "__main__":
    with open('Data/runs.pickle', 'rb') as handle:
        data = pickle.load(handle)

    summary_list = data['summary']
    config_list = data['config']
    name_list  = data['name']
    history_list = data['history']
    for i in range(len(data)):
        summary = summary_list[i]
        config = config_list[i]
        name = name_list[i]
        history = history_list[i]
        break

    observations = list(history["observation"])
    x_positions = []
    velocities = []
    r_intrinsic = []
    updates = []

    for i in range(len(observations)):
        if observations[i]:
            updates.append(list(history["num_update"])[i])
            for j in range(len(observations[i])):
                obs = observations[i][j]
                r_intrinsic.append(history["reward"][j])
                x_positions.append(obs[0])
                velocities.append(obs[1])
                print(i)
    
        
    print(x_positions)
    r_intrinsic_plot(r_intrinsic, x_positions, velocities, updates)

        

    # api = wandb.Api()
    # entity, project = "university-alberta", "ppo"
    # runs = api.runs(entity + "/" + project)


    # episodic_return = []
    # episode_number = []
    # min_episode_number = float('inf')
    # args = parse_args_visualize()

    # if not args.load:
    #     positions = []
    #     rewards = []
    #     for run in runs:
    #         exp_name = run.config['exp_name']
    #         gym_id = run.config['gym_id']
    #         agg_func = run.config['aggregate_function']
    #         num_steps = run.config['num_steps']
    #         if num_steps != 8:
    #             print(num_steps)
    #             continue
            
    #         # .summary contains output keys/values for
    #         # metrics such as accuracy.
    #         #  We call ._json_dict to omit large files
    #         df = run.history(samples=10000, keys=['observations'])
    #         i = 0
    #         if exp_name == args.exp_name and gym_id == "MountainCar-v0" :
    #             print(exp_name, run.config['seed'])
    #             position = df["observation"].to_numpy()
    #             positions.append(position)

    #             if "reward" in df.head():
    #                 reward = df["reward"].to_numpy()
    #                 rewards.append(reward)
    #                 print(reward)

                
    #     np.save('Data/observations__' + args.exp_name + "__" + agg_func, positions)
    #     np.save('Data/rewards__'  + args.exp_name + "__" + agg_func, rewards )

    # else:
    #     algorithms = {"PPO": "main_ppo", "PPO with count": "main_ppo_count"}
    #     # plt.bars(heatmap, bins)

    #     fig = plt.figure()
    #     grid = AxesGrid(fig, 111,
    #             nrows_ncols=(2, 1),
    #             axes_pad=0.5,
    #             share_all=True,
    #             label_mode="L",
    #             cbar_location="right",
    #             cbar_mode="single",
    #             )   
        
    #     i = 0
    #     for algo in algorithms.keys():
    #         exp_name = algorithms[algo]
    #         observations = np.load('Data/observations__' + exp_name + "__" +  'mean.npy', allow_pickle=True)
    #         rewards = np.load('Data/rewards__'  + exp_name + "__" + 'mean.npy', allow_pickle=True)
    #         x = []
    #         v = []
    #         for j in range(49):
    #             for observation in observations[j]:
    #                 if observation:
    #                     for o in observation:
    #                         x.append(o[0])
    #                         v.append(o[1])

    #         ax = grid[i]
    #         extent = [-1.2, 0.6, 0, 1]
    #         bins = np.linspace(extent[0], extent[1], num=50)
    #         heatmap, xedges = np.histogram(x, bins=bins)
    #         cax = ax.imshow(heatmap[np.newaxis,:], cmap="hot", aspect="auto", extent=extent)

    #         ax.set_yticks([])
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_title(algo)
    #         i += 1

    #     grid.cbar_axes[0].colorbar(cax)

    #     for cax in grid.cbar_axes:
    #         cax.toggle_label(True)



    #     # plt.pcolormesh(heatmap, cmap='Greys', shading='gouraud')
    #     plt.savefig('figures/MountainCar-v0-heatmap')
    #     plt.close()



    #     i = 0

    #     fig = plt.figure(figsize=(8, 10))
    #     grid = AxesGrid(fig, 111,
    #             nrows_ncols=(2, 1),
    #             axes_pad=0.5,
    #             share_all=True,
    #             label_mode="L",
    #             cbar_location="right",
    #             cbar_mode="single",
    #             )   
        

    #     for algo in algorithms.keys():
    #         exp_name = algorithms[algo]
    #         observations = np.load('Data/observations__' + exp_name + "__" +  'mean.npy', allow_pickle=True)
    #         rewards = np.load('Data/rewards__'  + exp_name + "__" + 'mean.npy', allow_pickle=True)
    #         x = []
    #         v = []
    #         for j in range(49):
    #             for observation in observations[j]:
    #                 if observation:
    #                     for o in observation:
    #                         x.append(o[0])
    #                         v.append(o[1])
    #         ax = grid[i]
    #         extent = [-1.2, 0.6, 0, 1]
    #         heatmap, _, _ = np.histogram2d(x, v, bins=50)
    #         print(heatmap.T)
    #         cax = ax.imshow(heatmap.T, cmap="hot", aspect='auto', extent=extent)

    #         ax.set_yticks([])
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_title(algo)
    #         i += 1

    #     grid.cbar_axes[0].colorbar(cax)

    #     for cax in grid.cbar_axes:
    #         cax.toggle_label(True)



    #     # plt.pcolormesh(heatmap, cmap='Greys', shading='gouraud')
    #     plt.savefig('figures/MountainCar-v0-position-velocity')
    #     plt.close()


    #     i = 0

    #     fig = plt.figure(figsize=(8, 10))
    #     grid = AxesGrid(fig, 111,
    #             nrows_ncols=(2, 1),
    #             axes_pad=0.5,
    #             share_all=True,
    #             label_mode="L",
    #             cbar_location="right",
    #             cbar_mode="single",
    #             )   
        

    #     for algo in algorithms.keys():
    #         exp_name = algorithms[algo]
    #         observations = np.load('Data/observations__' + exp_name + "__" +  'mean.npy', allow_pickle=True)
    #         rewards = np.load('Data/rewards__'  + exp_name + "__" + 'mean.npy', allow_pickle=True)
    #         x = []
    #         v = []
    #         for j in range(49):
    #             for observation in observations[j]:
    #                 if observation:
    #                     for o in observation:
    #                         x.append(o[0])
    #                         v.append(o[1])

    #         ax = grid[i]
    #         extent = [-1.2, 0.6, -1, 1]
    #         x = np.asarray(x)
    #         y = np.divide(-1*np.sin(3*x), 3)
    #         print(x[0], y[0])
    #         heatmap, _, _ = np.histogram2d( x, y,bins=50)
    #         cax = ax.imshow(heatmap.T, cmap="hot", aspect='auto', extent=extent)

    #         ax.set_yticks([])
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_title(algo)
    #         i += 1

    #     grid.cbar_axes[0].colorbar(cax)

    #     for cax in grid.cbar_axes:
    #         cax.toggle_label(True)



    #     # plt.pcolormesh(heatmap, cmap='Greys', shading='gouraud')
    #     plt.savefig('figures/MountainCar-v0-x-y')
    #     plt.close()


    #     fig = plt.figure(figsize=(8, 10))
    #     grid = AxesGrid(fig, 111,
    #             nrows_ncols=(1, 1),
    #             axes_pad=0.5,
    #             share_all=True,
    #             label_mode="L",
    #             cbar_location="right",
    #             cbar_mode="single",
    #             )   
        

    #     for algo in algorithms.keys():
    #         if algo == "PPO":
    #             continue
    #         exp_name = algorithms[algo]
    #         rewards = np.load('Data/rewards__'  + exp_name + "__" + 'mean.npy', allow_pickle=True)
    #         x = []
    #         v = []
    #         r = []
    #         for j in range(49):
    #             for observation in observations[j]:
    #                 if observation:
    #                     for o in observation:
    #                         x.append(o[0])
    #                         v.append(o[1]) 

    #         ax = grid[i]
    #         extent = [-1.2, 0.6, -1, 1]
    #         x = np.asarray(x)
    #         y = np.divide(-1*np.sin(3*x), 3)
    #         print(x[0], y[0])
    #         heatmap, _, _ = np.histogram2d( x, y,bins=50)
    #         cax = ax.imshow(heatmap.T, cmap="hot", aspect='auto', extent=extent)

    #         ax.set_yticks([])
    #         ax.set_xlim(extent[0], extent[1])
    #         ax.set_title(algo)
    #         i += 1

    #     grid.cbar_axes[0].colorbar(cax)

    #     for cax in grid.cbar_axes:
    #         cax.toggle_label(True)



    #     # plt.pcolormesh(heatmap, cmap='Greys', shading='gouraud')
    #     plt.savefig('figures/MountainCar-v0-x-y')
    #     plt.close()

        




