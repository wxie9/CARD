# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=Xformer



root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

# root_path_name=./dataset/ETT-small/
# data_path_name=ETTm2.csv
# model_id_name=ETTm2
# data_name=ETTm2

# root_path_name=./dataset/weather/
# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom

# root_path_name=./dataset/electricity/
# data_path_name=electricity.csv
# model_id_name=Electricity
# data_name=custom

#192 336 720

random_seed=2021

for pred_len in  192 336 720 96 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 2 \
      --d_model 16 \
      --d_ff 32 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 \
      --lradj Xformer --warmup_epochs 0 \
      2>&1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log \
      # &
done



# # export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_MODE=offline
# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
# seq_len=720
# model_name=Xformer

# root_path_name=./dataset/ETT-small/
# data_path_name=ETTh2.csv
# model_id_name=ETTh2
# data_name=ETTh2

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
#       --enc_in 7 \
#       --e_layers 6 \
#       --n_heads 8\
#       --d_model 128 \
#       --d_ff 256 \
#       --dropout 0.2\
#       --fc_dropout 0.2\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 400\
#       --patience 400 \
#       --itr 1 --batch_size 128 --learning_rate 0.0001 --lradj TST --use_multi_gpu --devices '0,1,2,3'
# done

# # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log      --same_smoothing --use_statistic 
# # \
#       # --lradj constant