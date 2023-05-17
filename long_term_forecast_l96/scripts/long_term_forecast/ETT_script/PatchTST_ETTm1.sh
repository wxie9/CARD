# export CUDA_VISIBLE_DEVICES=2

model_name=PatchTST


# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.2\
#   --train_epochs 100\
#   --patience 20\
#   --lradj TST \
#   --pct_start 0.4\
#   --itr 1 --batch_size 128 --learning_rate 0.0001 &
     
# done











model_name=PatchTST

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --train_epochs 100\
  --patience 20\
  --lradj TST \
  --pct_start 0.4\
  --itr 1 --batch_size 128 --learning_rate 0.0001 

export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --train_epochs 100\
  --patience 20\
  --lradj TST \
  --pct_start 0.4\
  --itr 1 --batch_size 128 --learning_rate 0.0001 


export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --train_epochs 100\
  --patience 20\
  --lradj TST \
  --pct_start 0.4\
  --itr 1 --batch_size 128 --learning_rate 0.0001 


export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --train_epochs 100\
  --patience 20\
  --lradj TST \
  --pct_start 0.4\
  --itr 1 --batch_size 128 --learning_rate 0.0001 