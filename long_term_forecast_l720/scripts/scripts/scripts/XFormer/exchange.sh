export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE=offline
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=Xformer

root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
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
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 3 \
      --n_heads 2 \
      --d_model 16 \
      --d_ff 32 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 8\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10 \
      --itr 1 --batch_size 8 --learning_rate 0.0001 \
      --lradj Xformer --momentum 5e-3 --dp_rank 8 --warmup_epochs 20 
done

# >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 