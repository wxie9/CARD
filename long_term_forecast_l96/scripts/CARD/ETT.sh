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
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
     --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001 \
    2>&1 | tee logs/LongForecasting/$model_name'_'ETTm1_96_$pred_len.log &\
    
done




for ((i = 0; i < 4; i++)) 
do 
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001 \
    2>&1 | tee logs/LongForecasting/$model_name'_'ETTm2_96_$pred_len.log &\
    
done






for ((i = 0; i < 4; i++)) 
do 
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001 \
    2>&1 | tee logs/LongForecasting/$model_name'_'ETTh1_96_$pred_len'.log' &\
    
done



for ((i = 0; i < 4; i++)) 
do 
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --batch_size 128 --learning_rate 0.0001 \
    2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_$pred_len'.log' \
    &

done





# # export CUDA_VISIBLE_DEVICES=6

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir -p ./logs/LongForecasting
# fi


# model_name=CARD

# for pred_len in 96 192 336 720
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_$pred_len \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --e_layers 2 \
#   --n_heads 2 \
#   --d_model 16 \
#   --d_ff 32 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj CARD \
#  --batch_size 128 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_$pred_len'.log' \
#     &


# done
