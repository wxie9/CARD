export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

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
#   --label_len 96 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 --seg_len 24 --learning_rate 1e-3 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj type1\

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
#   --label_len 96 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 --seg_len 24 --learning_rate 1e-3 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj type1\



# export CUDA_VISIBLE_DEVICES=7,6
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 96 \
#   --pred_len 336 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 --seg_len 24 --learning_rate 1e-3 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj type1 --use_multi_gpu --devices '0,1' &\


export CUDA_VISIBLE_DEVICES=5,4,3,2
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 --seg_len 24 --learning_rate 1e-3 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj type1 --use_multi_gpu --devices '0,1,2,3' &\