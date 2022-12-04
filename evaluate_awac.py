from d3rlpy.algos import AWAC
import argparse

# project-related dependencies
from lhd_env_nodes.tasks.reach_collision import LHDReachCollision
from evaluate import main


def evaluate_awac(config: dict):

    # create environment
    env = LHDReachCollision(max_episode_steps=config["max_episode_steps"],
                            success_reward=750,
                            collision_penalization=-1000,
                            dense_reward_weight=10,
                            publish_collision_points=False,
                            publish_info=True,
                            direction="forward",
                            verbose=False)

    # create model
    awac = AWAC(reward_scaler="standard",
                actor_encoder_factory="dense",
                critic_encoder_factory="dense",
                lam=1.0,
                n_action_samples=1,
                use_gpu=True)
    awac.build_with_env(env)
    awac.load_model(config["checkpoint"])

    main(env, awac, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval actor")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    evaluate_awac(config)
