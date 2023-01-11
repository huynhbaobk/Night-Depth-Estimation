python train.py \
    --batch_size 6 \
    --data_path "/media/aiteam/DataAI/depth_datasets/oxford/" \
    --log_dir "logs" \
    --split "oxford_night" \
    --png \
    --height 192 \
    --width 384 \
    --num_epochs 20 


python evaluate_depth.py --load_weights_folder /media/aiteam/DataAI/monodepth_night/monodepth2/logs/mono_oxford_day_512x256_batch8_epoch20/models/weights_4 \
                            --eval_split oxford_night_411 \
                            --eval_mono \
                            --png


python test_simple.py --image_path /media/aiteam/DataAI/depth_datasets/oxford/night_val_411 \
                        --output_path /media/aiteam/DataAI/depth_datasets/oxford/night_val_411_result \
                        --model_path /media/aiteam/DataAI/monodepth_night/monodepth2/logs/mono_oxford_day_512x256_batch8_epoch20/models/weights_4