import wandb
import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import AxesGrid
from PIL import Image

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


def _find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return np.unravel_index(idx, array.shape)


def r_intrinsic_plot(r_intrinsic, x_position, velocity, updates):
    images = []
   
    x = np.arange(-1.2, 0.6, 0.05)
    y = np.arange(-0.07, 0.07, 0.001)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)


    for i in range(1500):
       
        p = x_position[updates[i]: updates[i+1]]
        v = velocity[updates[i]: updates[i+1]]
        r = r_intrinsic[updates[i]: updates[i+1]]

        for j in range(len(p)):
            i_Y = _find_nearest_index(X, p[j])[1]
            i_X = _find_nearest_index(Y, v[j])[0]
            Z[i_X, i_Y] = r[j]

        # Plot the surface.
        if i % 5 != 0:
            continue
        print(i)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 

        ax.axes.set_xlim3d(left=-1.2, right=0.6) 
        ax.axes.set_ylim3d(bottom=-0.07, top=0.07) 
        ax.axes.set_zlim3d(bottom=-1, top=1) 

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False, vmax=1, vmin=-1)

                # Customize the z axis.
                # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
                # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

                # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        file_name = "figures/r_heatmap/3d/MountainCar-intrinsic-reward" + "_" + str(i) + '.png'
        ax.view_init(10, 60)
        plt.savefig(file_name)
        images.append(Image.open(file_name))
        plt.close()
        
    plt.show()
    images[0].save('figures/r_heatmap/3d/main.gif', save_all=True, append_images=images, duration=200, loop=0)

def r_intrinsic_heatmap(r_intrinsic, x_position, velocity, updates):
    images = []
   
    x = np.arange(-1.2, 0.6, 0.05)
    y = np.arange(-0.07, 0.07, 0.005)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    levels = np.linspace(-1, 1, 20)
    x = X.flatten()
    y = Y.flatten()

    for i in range(1300):
        p = x_position[updates[i]: updates[i+1]]
        v = velocity[updates[i]: updates[i+1]]
        r = r_intrinsic[updates[i]: updates[i+1]]

        for j in range(len(p)):
            if np.isnan(r[j]):
                continue
            i_X = _find_nearest_index(X, p[j])[1]
            i_Y = _find_nearest_index(Y, v[j])[0]
            Z[i_X, i_Y] = r[j]
        
        # Plot the surface.
        if i % 5 != 0:
            continue
        print(i)
        fig, ax = plt.subplots() 

        
        ax.axes.set_xlim(left=-1.2, right=0.6) 
        ax.axes.set_ylim(bottom=-0.07, top=0.07) 
        extent = [-1.2, 0.6, -0.07, 0.07]
        surf = ax.tricontourf(x, y, Z.flatten(), levels=levels)
                # Customize the z axis.
                # ax.set_zlim(-1.01, 1.01)
       
                # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        file_name = "figures/r_heatmap/2d/MountainCar-intrinsic-reward" + "_" + str(i) + '.png'
        # ax.view_init(10, 60)
        plt.savefig(file_name)
        images.append(Image.open(file_name))
        plt.close()

    images[0].save('figures/r_heatmap/2d/main.gif', save_all=True,append_images=images, duration=200, loop=0)


def observation_heatmap(x_position, velocity, updates):
    images = []

    for i in range(1300):
        print(i)
        p = x_position[0: updates[i+1]]
        v = velocity[0: updates[i+1]]


        fig, ax = plt.subplots() 
        extent = [-1.2, 0.6, -0.07, 0.07]
        heatmap, _, _ = np.histogram2d(p, v, bins=70, range=[[-1.2, 0.6], [-0.07, 0.07]])
        heatmap = ax.imshow(heatmap.T, cmap="hot", aspect='auto', extent=extent, vmin=0, vmax=1000)

        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax.set_title("PPO with counts")
        i += 1
        
        fig.colorbar(heatmap, ax=ax, extend='both', aspect=5)
        file_name = "figures/observations_heatmap/position_velocity/MountainCar-intrinsic-reward" + "_" + str(i) + '.png'
        # ax.view_init(10, 60)
        plt.savefig(file_name)
        images.append(Image.open(file_name))
        plt.close()
        
    images[0].save('figures/observations_heatmap/position_velocity/main.gif', save_all=True,append_images=images, duration=200, loop=0)



def position_heatmap():
    pass
        

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

    x_positions = np.array(history['x_position'])
    velocities = np.array(history['velocity'])
    r_intrinsic = np.array(history['reward'])
    updates = np.array(history['num_update'])
    updates = np.where(updates[:-1] != updates[1:])[0]
    updates = np.insert(updates, 0, 0)
    
    # r_intrinsic_plot(r_intrinsic, x_positions, velocities, updates)
    r_intrinsic_heatmap(r_intrinsic, x_positions, velocities, updates)
    observation_heatmap(x_positions, velocities, updates)







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

        




