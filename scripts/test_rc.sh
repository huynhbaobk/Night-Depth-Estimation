#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_robotcar_disp.py night steps_rc checkpoints/steps_rc/rc_best.ckpt --test 1 --vis 1
cd evaluation
python eval_robotcar.py night

# done