from d3rlpy.algos import IQL
import argparse
from pathlib import Path

# project-related dependencies
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision
from evaluate import eval


def evaluate_iql(config: dict):
    env = LHDReachCollision(max_episode_steps=config["max_episode_steps"],
                            success_reward=100,
                            collision_penalization=-500,
                            dense_reward_weight=1,
                            publish_collision_points=False,
                            publish_info=True,
                            direction='forward',
                            target_search_limit=300,
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

    export_folder = str(Path(config["checkpoint"]).parents[0])
    eval(env, iql, config, export_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    evaluate_iql(config)
