#!/bin/bash

python test_robotcar_disp.py day mono2_rc_day checkpoints/mono2_rc_day/checkpoint_epoch=19.ckpt --test 1 --vis 0
cd evaluation
python eval_robotcar.py day

# done