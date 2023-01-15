#!/bin/bash

run_buffer_1="d3rlpy_logs/IQL_20230107144029/model_70100.pt"
run_buffer_2="d3rlpy_logs/IQL_20230107150907/model_70300.pt"
run_buffer_3="d3rlpy_logs/IQL_20230107153453/model_35150.pt"
run_buffer_4="d3rlpy_logs/IQL_20230108122114/model_35100.pt"
python evaluate_iql.py --checkpoint $run_buffer_1 --num_eval_episodes 500 --plot
python evaluate_iql.py --checkpoint $run_buffer_2 --num_eval_episodes 500 --plot
python evaluate_iql.py --checkpoint $run_buffer_3 --num_eval_episodes 500 --plot
python evaluate_iql.py --checkpoint $run_buffer_4 --num_eval_episodes 500 --plot