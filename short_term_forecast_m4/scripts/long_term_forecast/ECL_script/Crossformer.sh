export CUDA_VISIBLE_DEVICES=7

model_name=Crossformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 --seg_len 24 --learning_rate 5e-5 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj constant &\



export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 --seg_len 24 --learning_rate 5e-5 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj constant &\


export CUDA_VISIBLE_DEVICES=4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 --seg_len 24 --learning_rate 5e-5 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj constant &\


export CUDA_VISIBLE_DEVICES=5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 --seg_len 24 --learning_rate 5e-5 --d_model 64 --d_ff 128 --n_heads 2 --dropout 0.2 --batch_size 32 --lradj constant &\