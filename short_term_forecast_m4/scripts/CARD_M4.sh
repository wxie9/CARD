if [ ! -d "./logs/ShortForecasting/" ]; then
mkdir -p ./logs/ShortForecasting/
fi



# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY=
export WANDB_MODE=offline

model_name=CARD


export CUDA_VISIBLE_DEVICES=0
python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Monthly' \
--model_id m4_layer_2_Monthly \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--dropout 0.0 \
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 16 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_Monthly.log \
&



export CUDA_VISIBLE_DEVICES=1
python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Yearly' \
--model_id m4_layer_2Yearly \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--dropout 0.0 \
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 3 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Yearly.log &


export CUDA_VISIBLE_DEVICES=2

python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Quarterly' \
--model_id m4_layer_2_Quarterly \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--dropout 0.0 \
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 4 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Quarterly.log &

export CUDA_VISIBLE_DEVICES=3

python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Daily' \
--model_id m4_layer_2_Daily \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--dropout 0.0 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 16 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Daily.log &




python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Weekly' \
--model_id m4_layer_2_Weekly \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--dropout 0.0 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 16 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Weekly.log &




python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Hourly' \
--model_id m4_layer_2_Hourly \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 16 \
--patience 400 \
--train_epochs 100 \
--stride 1 \
--n_heads 16 \
--dropout 0.0 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Hourly.log &


