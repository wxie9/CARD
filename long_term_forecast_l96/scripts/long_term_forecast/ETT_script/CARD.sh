
# export CUDA_VISIBLE_DEVICES=6

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi


model_name=CARD

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
#   --d_ff 128 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj CARD \
#  --batch_size 128 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_$pred_len'.log' \
    
# done


for pred_len in 720 96 192 336 
do

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
  --d_ff 128 \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --patience 20\
   --train_epochs 100 --lradj CARD \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
   --dp_rank 8 --top_k 5   --mask_rate 0 --warmup_epochs 0 \
    2>&1 | tee logs/LongForecasting/$model_name'_'ETTm1_96_$pred_len.log 

done


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
#   --n_heads 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj Xformer \
#  --batch_size 32 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_$pred_len'.log' \
    
# done


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#  --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --e_layers 3 \
#   --n_heads 2 \
#   --d_model 16 \
#   --d_ff 32 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj Xformer \
#  --batch_size 128 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_192.log &


   
# # export CUDA_VISIBLE_DEVICES=2
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#  --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --e_layers 3 \
#   --n_heads 2 \
#   --d_model 16 \
#   --d_ff 32 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj Xformer \
#  --batch_size 128 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_336.log &

   
# #  export CUDA_VISIBLE_DEVICES=3  
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#  --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --e_layers 3 \
#   --n_heads 2 \
#   --d_model 16 \
#   --d_ff 32 \
#   --dropout 0.3\
#   --fc_dropout 0.3\
#   --head_dropout 0\
#   --patch_len 16\
#   --stride 8\
#     --patience 20\
#    --train_epochs 100 --lradj Xformer \
#  --batch_size 128 --learning_rate 0.0001 \
#     2>&1 | tee logs/LongForecasting/$model_name'_'ETTh2_96_720.log &

   
   