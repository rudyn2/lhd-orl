#!/bin/bash
run_iql="d3rlpy_logs/IQL_20230108010603/model_87850.pt"
run_cql="d3rlpy_logs/CQL_20230108012353/model_75250.pt"
run_crr="d3rlpy_logs/CRR_20230108021339/model_67350.pt"
run_awac="d3rlpy_logs/AWAC_20230108022528/model_79900.pt"
python evaluate_iql.py --checkpoint $run_iql --num_eval_episodes 500 --plot
python evaluate_cql.py --checkpoint $run_cql --num_eval_episodes 500 --plot
python evaluate_crr.py --checkpoint $run_crr --num_eval_episodes 500 --plot
python evaluate_awac.py --checkpoint $run_awac --num_eval_episodes 500 --plot