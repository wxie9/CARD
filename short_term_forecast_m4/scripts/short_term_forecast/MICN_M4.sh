export CUDA_VISIBLE_DEVICES=4

model_name=MICN

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 15 \
  --patience 15 \
  --loss 'SMAPE' --d_model 512 --d_ff 2048 --learning_rate 0.001 --lradj type1 & \