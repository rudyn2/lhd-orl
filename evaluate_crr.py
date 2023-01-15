from d3rlpy.algos import CRR
import argparse
from custom_factories import CustomDenseEncoder
from pathlib import Path

# project-related dependencies
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision
from evaluate import eval


def evaluate_crr(config: dict):

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
    crr = CRR(reward_scaler="standard",
              actor_encoder_factory=CustomDenseEncoder(hidden_units=[512, 512, 512]),
              critic_encoder_factory=CustomDenseEncoder(hidden_units=[512, 512, 512]),
              advantage_type="mean",
              weight_type="exp",
              n_critics=1,
              use_gpu=True)
    crr.build_with_env(env)
    crr.load_model(config["checkpoint"])

    export_folder = str(Path(config["checkpoint"]).parents[0])
    eval(env, crr, config, export_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    evaluate_crr(config)
