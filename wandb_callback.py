import wandb
from d3rlpy.base import LearnableBase


def wandb_callback(algo: LearnableBase, epoch: int, total_step: int):
    to_log = dict(epoch=epoch, step=total_step)
    for key in algo.active_logger._metrics_buffer.keys():
        to_log[key] = algo.active_logger._metrics_buffer[key][-1]
    wandb.log(to_log)