if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi


# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY=
export WANDB_MODE=offline

model_name=CARD
pred_lens=(96 192 336 720)
cuda_ids1=(0 1 2 3)



for ((i = 0; i < 4; i++)) 
do 
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --itr 1 --batch_size 128 --learning_rate 0.0001 \
    --dp_rank 8 --top_k 5   --mask_rate 0 --warmup_epochs 0 \
    2>&1 | tee logs/LongForecasting/$model_name'_'Weather_96_$pred_len'.log' & \


done
