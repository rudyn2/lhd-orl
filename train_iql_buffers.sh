#!/bin/bash
# buffer_1="data/buffer_1.pkl"
# buffer_2="data/buffer_2.pkl"
# buffer_3="data/buffer_3.pkl"
buffer_4="data/buffer_4_1.pkl"
# python run_iql.py --data $buffer_1 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
# python run_iql.py --data $buffer_2 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
# python run_iql.py --data $buffer_3 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50
python run_iql.py --data $buffer_4 --encoder_type=dense --expectile=0.9 --max_weight=50 --weight_temp=10 --batch_size 256 --n_epochs 50