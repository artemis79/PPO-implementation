
import os
import random
import time
from distutils.util import strtobool
from logger import Logger

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, make_env
from utils import IHT, tiles
from agents.agent import Agent


def get_tiles(iht, observations, tiling_num, tiling_size, scale):
    features = np.zeros((len(observations), tiling_num * tiling_size))
    for i in range(len(observations)):
        obs = observations[i]
        indices = tiles(iht, tiling_num, obs * scale)
        one_hot = np.zeros((len(indices), tiling_size), dtype=int)
        one_hot[np.arange(len(indices)),indices] = 1
        one_hot = one_hot.flatten()
        features[i] = one_hot
    
    return features



# Think abou the shapes of the vectors
def intrinsic_reward(counts, feature, action, action_space_size, aggregate_function, beta=1):
    n_s = []
    n_a = 0
    for a in range(action_space_size):
        count_a = counts[:, a].flatten() * feature
        if aggregate_function == 'min':
            count_a = np.min(count_a[np.nonzero(count_a)])
        elif aggregate_function == 'mean':
            count_a = np.mean(count_a[np.nonzero(count_a)])
        else:
            print("not valid aggregate function")
            print("choose min")
            count_a = np.min(count_a[np.nonzero(count_a)])

        n_s.append(count_a)
        if a == action:
            n_a = count_a

    n_s = np.sum(n_s)
    return beta * np.sqrt(2*np.log(n_s)/n_a)



if __name__ == "__main__":

    args = parse_args()
    args.exp_name = os.path.basename(__file__).rstrip(".py")
    # print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    # Encode args as text
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    hyperparameters = "|".join([f"{key}={value}" for key, value in vars(args).items()])
    print(hyperparameters)
    # logger = Logger(hyperparameters)



    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)


    # TRY NOT TO MODIFY: start the game
    episode_number = 0
    global_step = 0
    start_time = time.time()
    observation, info = envs.reset()
    next_obs = torch.Tensor(observation).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    max_return = float('-inf')

    # Count setting
    num_tiling = 10
    tile_size = 1000
    num_tiles = [4, 4]
    observation_high = envs.observation_space.high[0]
    # observation_high[observation_high == float('inf')] = num_tiles
    observation_low = envs.observation_space.low[0]
    # observation_low[observation_low == float('-inf')] = 0
    scale = num_tiles / (observation_high - observation_low)
    iht = IHT(tile_size)
    action_space_size = envs.action_space.shape[0]
    counts = np.ones((num_tiling*tile_size, action_space_size)) * args.count_start
    aggregate_function = args.aggregate_function
    beta = args.beta
    total_compute_steps = 0

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        mid_counts = np.zeros((args.num_envs, num_tiling*tile_size, action_space_size))
        # print("=======================")
        # print("Counts")
        # print(counts)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        returns = np.zeros(args.num_envs)

        for step in range(0, args.num_steps):
            
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action


            # Add to count
            if args.count:
                features = get_tiles(iht, obs[step], num_tiling, tile_size, scale)
                for i in range(args.num_envs):
                    mid_counts[i, :, action[i]] +=  features[i]
            

            # print(actions)
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

            if args.count:
                intrinsic_rewards = []
                for i in range(args.num_envs):
                    if args.update_counts_step:
                        intrinsic_rewards.append(intrinsic_reward(counts+mid_counts[i], features[i], action[i], action_space_size, aggregate_function, beta))
                    else:
                        intrinsic_rewards.append(intrinsic_reward(counts, features[i], action[i], action_space_size, aggregate_function, beta))

                reward = reward + intrinsic_rewards
                    
            returns = returns + reward        
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            


            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)
            
            i_environment = 0
            for key, items in info.items():
                if key=='final_info':
                    for item in items:
                        if item and "episode" in item.keys():
                            # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_return_per_episode", item["episode"]["r"], episode_number)
                            writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                            writer.add_scalar("charts/episodic_length_per_episode", item["episode"]["l"], episode_number)
                            max_return = max(max_return, item["episode"]["r"])
                            total_compute_steps += item["episode"]["l"][0]
                            # Log episodic return to 
                            # logger.log_episode_return([item["episode"]["r"][0], total_compute_steps, episode_number])

                            if args.track:
                                run.log({"episodic_return": item["episode"]["r"],
                                        "episode_length": item["episode"]["l"],
                                        "global_step": global_step,}
                                        )
                    episode_number += 1

        # Add counts from rollouts to main count

        counts += np.sum(mid_counts, axis=0)
        if args.track:
            run.log({"Return_variance": np.var(returns),
                     "num_update": update
                     })

        print(returns)
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_rewards = rewards.reshape(-1,)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Log observations and rewards
        # logger.log_observation(b_obs, update)
        # logger.log_rewards(b_rewards, update)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track:
            run.log({"charts/learning_rate": optimizer.param_groups[0]["lr"], "global_step": global_step})
            run.log({"losses/value_loss": v_loss.item(), "global_step": global_step})
            run.log({"losses/policy_loss": pg_loss.item(),"global_step": global_step})
            run.log({"losses/entropy": entropy_loss.item(),"global_step": global_step})
            run.log({"losses/old_approx_kl": old_approx_kl.item(),"global_step": global_step})
            run.log({"losses/approx_kl": approx_kl.item(),"global_step": global_step})
            run.log({"losses/clipfrac": np.mean(clipfracs), "global_step": global_step})
            run.log({"losses/explained_variance": explained_var, "global_step": global_step})
            run.log({"charts/SPS": int(global_step / (time.time() - start_time)), "global_step": global_step})
        

    envs.close()
    writer.close() 
    



