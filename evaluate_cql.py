from d3rlpy.algos import CQL
import argparse
from pathlib import Path

# project-related dependencies
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision
from evaluate import eval


def evaluate_cql(config: dict):

    # create environment
    env = LHDReachCollision(max_episode_steps=config["max_episode_steps"],
                            success_reward=100,
                            collision_penalization=-500,
                            dense_reward_weight=1,
                            publish_collision_points=False,
                            publish_info=True,
                            direction="forward",
                            verbose=False)

    # create model
    cql = CQL(reward_scaler="standard",
              actor_encoder_factory="dense",
              critic_encoder_factory="dense",
              alpha_threshold=10.0,
              conservative_weight=1.0,
              soft_q_backup=False,
              n_critics=2,
              use_gpu=True)
    cql.build_with_env(env)
    cql.load_model(config["checkpoint"])

    export_folder = str(Path(config["checkpoint"]).parents[0])
    eval(env, cql, config, export_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    evaluate_cql(config)
