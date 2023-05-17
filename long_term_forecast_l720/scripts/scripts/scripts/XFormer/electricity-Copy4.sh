# export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=offline

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=Xformer

# root_path_name=./dataset/ETT-small/
# data_path_name=ETTm2.csv
# model_id_name=ETTm2
# data_name=ETTm2

# root_path_name=./dataset/weather/
# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for pred_len in 720 336 192 96

# export CUDA_VISIBLE_DEVICES=0,1 

# pred_len=336
# export CUDA_VISIBLE_DEVICES=6
do
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 0 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff  256 \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 100\
  --patience 100 \
  --itr 1 --batch_size 32 --dp_rank 8 --learning_rate 0.0001 \
  --lradj Xformer --warmup_epochs 20 --use_multi_gpu --devices '0,1,2,3' 

done
# export CUDA_VISIBLE_DEVICES=2,3
# pred_len=192
# # export CUDA_VISIBLE_DEVICES=5
# # do
# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --data_path $data_path_name \
#   --model_id $model_id_name'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data $data_name \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 321 \
#   --e_layers 3 \
#   --n_heads 4 \
#   --d_model 32 \
#   --d_ff  128\
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 8\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 100 \
#   --itr 1 --batch_size 32 --dp_rank 8 --learning_rate 0.0001 \
#   --lradj Xformer --warmup_epochs 20 --use_multi_gpu --devices '0,1' &


# export CUDA_VISIBLE_DEVICES=4,5
# pred_len=96
# # export CUDA_VISIBLE_DEVICES=4
# # do
# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --data_path $data_path_name \
#   --model_id $model_id_name'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data $data_name \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 321 \
#   --e_layers 3 \
#   --n_heads 4 \
#   --d_model 32 \
#   --d_ff  128\
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 8\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 100 \
#   --itr 1 --batch_size 32 --dp_rank 8 --learning_rate 0.0001 \
#   --lradj Xformer --warmup_epochs 20 --use_multi_gpu --devices '0,1' &


# export CUDA_VISIBLE_DEVICES=6,7
# pred_len=720
# # export CUDA_VISIBLE_DEVICES=7
# # do
# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --data_path $data_path_name \
#   --model_id $model_id_name'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data $data_name \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 321 \
#   --e_layers 3 \
#   --n_heads 4 \
#   --d_model 32 \
#   --d_ff  128\
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 8\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 100 \
#   --itr 1 --batch_size 32 --dp_rank 8 --learning_rate 0.0001 \
#   --lradj Xformer --warmup_epochs 20 --use_multi_gpu --devices '0,1' &



# done



# export CUDA_VISIBLE_DEVICES=7
# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
# seq_len=720
# model_name=Xformer

# root_path_name=./dataset/weather/
# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom

# random_seed=2021
# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name_$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 32 \
#       --d_ff 32 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#     --lradj constant\
#       --train_epochs 100\
#       --patience 20\
#       --itr 1 --batch_size 32 --learning_rate 0.0001
# done
#  # --use_multi_gpu --devices '0,1,2,3'


# # # export CUDA_VISIBLE_DEVICES=4,5,6,7
# # # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # if [ ! -d "./logs" ]; then
# #     mkdir ./logs
# # fi

# # if [ ! -d "./logs/LongForecasting" ]; then
# #     mkdir ./logs/LongForecasting
# # fi
# # seq_len=720
# # model_name=Xformer

# # root_path_name=./dataset/electricity/
# # data_path_name=electricity.csv
# # model_id_name=Electricity
# # data_name=custom

# # random_seed=2021
# # for pred_len in 96 192 336 720
# # do
# #     python -u run_longExp.py \
# #       --random_seed $random_seed \
# #       --is_training 1 \
# #       --root_path $root_path_name \
# #       --data_path $data_path_name \
# #       --model_id $model_id_name_$seq_len'_'$pred_len \
# #       --model $model_name \
# #       --data $data_name \
# #       --features M \
# #       --seq_len $seq_len \
# #       --pred_len $pred_len \
# #       --enc_in 321 \
# #       --e_layers 2 \
# #       --n_heads 16 \
# #       --d_model 128 \
# #       --d_ff 256 \
# #       --dropout 0.2\
# #       --fc_dropout 0.2\
# #       --head_dropout 0\
# #       --patch_len 16\
# #       --stride 8\
# #       --des 'Exp' \
# #       --train_epochs 100\
# #       --patience 100\
# #       --lradj TST\
# #       --pct_start 0.2\
# #       --itr 1 --batch_size 8 --learning_rate 0.0001 --dp_rank 32 --use_multi_gpu --devices '0,1,2,3,4,5,6,7'
# # done
# # --dp_rank 8 '


# # export CUDA_VISIBLE_DEVICES=4,5,6,7
# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
# seq_len=720
# model_name=Xformer

# root_path_name=./dataset/electricity/
# data_path_name=electricity.csv
# model_id_name=Electricity
# data_name=custom

# random_seed=2021
# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name_$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 2 \
#       --n_heads 4 \
#       --d_model 32 \
#       --d_ff 32 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 400\
#       --patience 400 \
#       --itr 1 --batch_size 32 --learning_rate 0.0001 \
#       --lradj TST  --use_multi_gpu --devices '0,1,2,3'
# done