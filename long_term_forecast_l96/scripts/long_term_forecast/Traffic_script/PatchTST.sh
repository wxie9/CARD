# export CUDA_VISIBLE_DEVICES=7,6,5,4

model_name=PatchTST


# for pred_len in 96 192 336 720
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --e_layers 3 \
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.2\
#   --fc_dropout 0.2\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 10\
#   --lradj 'TST'\
#   --pct_start 0.2\
#   --itr 1 --batch_size 24 --learning_rate 0.0001 --use_multi_gpu --devices '0,1,2,3'

# done

# export CUDA_VISIBLE_DEVICES=7

model_name=PatchTST

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --e_layers 3 \
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.2\
#   --fc_dropout 0.2\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 10\
#   --lradj 'TST'\
#   --pct_start 0.2\
#   --itr 1 --batch_size 24 --learning_rate 0.0001 --use_multi_gpu --devices '0,1' \ 
#   &



# export CUDA_VISIBLE_DEVICES=7,6
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --e_layers 3 \
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.2\
#   --fc_dropout 0.2\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 10\
#   --lradj 'TST'\
#   --pct_start 0.2\
#   --itr 1 --batch_size 24 --learning_rate 0.0001  --use_multi_gpu --devices '0,1' \
#   &


export CUDA_VISIBLE_DEVICES=1,2
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 100\
  --patience 10\
  --lradj 'TST'\
  --pct_start 0.2\
  --itr 1 --batch_size 24 --learning_rate 0.0001  --use_multi_gpu --devices '0,1' \
  &


# export CUDA_VISIBLE_DEVICES=2
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --e_layers 3 \
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.2\
#   --fc_dropout 0.2\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#   --des 'Exp' \
#   --train_epochs 100\
#   --patience 10\
#   --lradj 'TST'\
#   --pct_start 0.2\
#   --itr 1 --batch_size 24 --learning_rate 0.0001  --use_multi_gpu --devices '0,1' \
#   &