from d3rlpy.algos import CQL
import argparse
import numpy as np
import sys

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
                            verbose=False)
    
    # create model
    cql = CQL(use_gpu=True)
    cql.build_with_env(env)
    cql.load_model(config["checkpoint"])

    # evaluate
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    speeds = []
    episodes = config["num_eval_episodes"]
    success = []

    for episode in range(episodes):
        ep_observations = []
        ep_actions = []
        ep_rewards = []
        ep_next_observations = []
        ep_dones = []
        ep_speeds = []
        ep_success = False
        observation = env.reset()
        current_step = 0
        while True:
            # perform 1-step in the environment
            action = cql.predict([observation])[0]
            next_observation, reward, done, info = env.step(action)

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
            current_step += 1

            if done:
                observations.append(ep_observations)
                actions.append(ep_actions)
                rewards.append(ep_rewards)
                next_observations.append(ep_next_observations)
                dones.append(ep_dones)
                speeds.append(ep_speeds)
                success.append(ep_success)
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

    print("-" * 50)
    print("{:<25} {:<10}".format('Key', 'Value'))
    for k, v in to_log.items():
        print("{:<25} {:<10}".format(k, v))
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str,
                        default="/home/rudy/lhd_gazebo_ws/src/lhd_navigation_ml/lhd_navigation_rl/src/actor_last.pth")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)

    args = parser.parse_args()
    config = vars(args)
    main(config)


