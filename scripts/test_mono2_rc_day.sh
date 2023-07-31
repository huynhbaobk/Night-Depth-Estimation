#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_robotcar_disp.py day mono2_rc_day checkpoints/mono2_rc_day/checkpoint_epoch=5.ckpt --test 1 --vis 1
cd evaluation
python eval_robotcar.py day

# done