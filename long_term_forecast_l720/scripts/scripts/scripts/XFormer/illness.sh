export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Xformer

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
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
            --patience 100\
      --lradj Xformer\
      --itr 1 --batch_size 16 --learning_rate 0.0001  --warmup_epochs 20 
       # --momentum 5e-3
done