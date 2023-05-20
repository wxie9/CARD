if [ ! -d "./logs/ShortForecasting/" ]; then
mkdir -p ./logs/ShortForecasting/
fi



export CUDA_VISIBLE_DEVICES=1
model_name=CARD



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
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 256 \
--patch_len 16\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_Monthly.log \
&




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
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 256 \
--patch_len 3\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Yearly.log &




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
--factor 3 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 256 \
--patch_len 4\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Quarterly.log &



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
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 256 \
--patch_len 16\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
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
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 256 \
--patch_len 16\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
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
--d_ff 256 \
--patch_len 16\
--patience 400\
--train_epochs 100\
--stride 1\
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0\
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Hourly.log &


