from d3rlpy.algos import AlgoBase
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import os


# project-related dependencies
from lhd_env_nodes.lhd_env import LHDEnv


BACKGROUND_IMG_PATH = "/home/rudy/lhd_ws/src/ghh_slp_14h_navigation_ml/lhd_gazebo_env/resources/maps/key_map-skeleton.png"


def calculate_stats(metric_list: list, name: str):
    return {
        f"{name}/mean": np.mean(metric_list),
        f"{name}/std": np.std(metric_list),
        f"{name}/min": np.min(metric_list),
        f"{name}/max": np.max(metric_list)
    }


def eval(env: LHDEnv, algo: AlgoBase, config: dict, export_folder: str):

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    speeds = []
    episodes = config["num_eval_episodes"]
    success = []
    collisions = []
    time_outs = []
    positions, start_positions, target_positions, end_positions = [], [], [], []

    for episode in range(episodes):
        ep_observations = []
        ep_actions = []
        ep_rewards = []
        ep_next_observations = []
        ep_dones = []
        ep_speeds = []
        ep_positions = []
        ep_success, ep_timeout = False, False
        observation = env.reset()
        start_positions.append(env.get_position())
        target_positions.append(env.get_target_position())
        current_step = 0
        while True:
            # perform 1-step in the environment
            action = algo.predict([observation])[0]
            next_observation, reward, done, info = env.step(action)
            ep_positions.append(env.get_position())

            sys.stdout.write(
                f"Evaluating. Episode: {episode}/{episodes} step={current_step}/{env.get_max_episode_steps()}"
                f", act=({action[0]:.2f}, {action[1]:.2f})\r")

            ep_observations.append(observation)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_next_observations.append(next_observation)
            ep_speeds.append(info["speed_mag"])
            observation = next_observation
            ep_success = ep_success or info['success']
            ep_timeout = ep_timeout or info['time_out']
            current_step += 1

            if done:
                end_positions.append(env.get_position())
                observations.append(ep_observations)
                actions.append(ep_actions)
                rewards.append(ep_rewards)
                next_observations.append(ep_next_observations)
                dones.append(ep_dones)
                speeds.append(ep_speeds)
                success.append(ep_success)
                collisions.append(info['collision'])
                time_outs.append(ep_timeout)
                positions.append(ep_positions)
                break
    env.close()

    results = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        speeds=speeds
    )
    returns = [np.sum(e) for e in results["rewards"]]
    mean_returns = [np.mean(e) for e in results["rewards"]]
    steps = [len(e) for e in results["rewards"]]
    speeds = [np.mean(e) for e in results["speeds"]]

    to_log = dict()
    to_log["episodes"] = len(returns)
    to_log.update(calculate_stats(steps, name="steps"))
    to_log.update(calculate_stats(returns, name="return"))
    to_log.update(calculate_stats(mean_returns, name="mean_return"))
    to_log.update(calculate_stats(speeds, name="speed"))
    to_log["success_rate"] = np.sum(success) / len(success)
    to_log["collisions"] = np.sum(collisions)
    to_log["time_outs"] = np.sum(time_outs)

    date_as_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_name = f"eval_results_{date_as_str}"

    with open(os.path.join(export_folder, f"{eval_name}.txt"), "w") as f:
        f.write("-" * 50 + "\n")
        f.write("{:<25} {:<10}\n".format('Key', 'Value'))
        for k, v in to_log.items():
            f.write("{:<25} {:<10}\n".format(k, v))

    if config["plot"]:
        fig, ax = plt.subplots(figsize=(22, 14))
        background = plt.imread(BACKGROUND_IMG_PATH)
        ax.imshow(background, extent=[-60, 60, -40, 40])

        episode_number = 0
        for trajectory, start_position, target_position, end_position in zip(positions, start_positions,
                                                                             target_positions, end_positions):
            trajectory = np.array(trajectory)
            has_collision = collisions[episode_number]
            has_timeout = time_outs[episode_number]
            trajectory_color = "r" if has_collision else "g"
            trajectory_color = "b" if has_timeout else trajectory_color
            timeout_tag = "*" if has_timeout else ""
            if has_collision or has_timeout:
                ax.plot([target_position[0]], [target_position[1]], "x", label=f"Target_{episode_number}{timeout_tag}")
                ax.plot([start_position[0]], [start_position[1]], "o", label=f"Start_{episode_number}{timeout_tag}")
                ax.plot([end_position[0]], [end_position[1]], "*", label=f"End_{episode_number}{timeout_tag}")
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=trajectory_color)
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=trajectory_color, alpha=0.35)
            episode_number += 1

        ax.legend()
        plt.savefig(os.path.join(export_folder, f"{eval_name}.png"))

    print(f"Results saved at {os.path.join(export_folder, f'{eval_name}.txt')}")
