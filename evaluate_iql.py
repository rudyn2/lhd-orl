from d3rlpy.algos import IQL
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt


# project-related dependencies
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision



def calculate_stats(metric_list: list, name: str):
    return {
        f"{name}/mean": np.mean(metric_list),
        f"{name}/std": np.std(metric_list),
        f"{name}/min": np.min(metric_list),
        f"{name}/max": np.max(metric_list)
    }


def main(config: dict):

    # create environment
    env = LHDReachCollision(max_episode_steps=config["max_episode_steps"],
                            success_reward=750,
                            collision_penalization=-1000,
                            dense_reward_weight=10,
                            publish_collision_points=False,
                            publish_info=True,
                            only_forward=True,
                            verbose=False)
    
    # create model
    iql = IQL(reward_scaler="standard",
              actor_encoder_factory="dense",
              critic_encoder_factory="dense",
              max_weight=50,
              weight_temp=10,
              expectile=0.9,
              use_gpu=True)
    iql.build_with_env(env)
    iql.load_model(config["checkpoint"])

    # evaluate
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
            action = iql.predict([observation])[0]
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
    to_log.update(calculate_stats(steps, name="steps"))
    to_log.update(calculate_stats(returns, name="return"))
    to_log.update(calculate_stats(mean_returns, name="mean_return"))
    to_log.update(calculate_stats(speeds, name="speed"))
    to_log["success_rate"] = np.sum(success) / len(success)
    to_log["collisions"] = np.sum(collisions)
    to_log["time_outs"] = np.sum(time_outs)

    print("-" * 50)
    print("{:<25} {:<10}".format('Key', 'Value'))
    for k, v in to_log.items():
        print("{:<25} {:<10}".format(k, v))
    print("-" * 50)
    
    if config["plot"]:
        fig, ax = plt.subplots(figsize=(22, 14))
        background = plt.imread("/home/rudy/lhd_gazebo_ws/src/lhd_navigation_ml/lhd_gazebo_env/"
                                "resources/maps/key_map-skeleton.png")
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
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str,
                        default="/home/rudy/lhd_gazebo_ws/src/lhd_navigation_ml/lhd_navigation_rl/src/actor_last.pth")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    main(config)


