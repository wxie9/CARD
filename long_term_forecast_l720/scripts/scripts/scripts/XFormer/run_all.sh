export CUDA_VISIBLE_DEVICES=0
bash ./scripts/scripts/scripts/XFormer/etth2.sh &

export CUDA_VISIBLE_DEVICES=1
bash ./scripts/scripts/scripts/XFormer/etth1.sh 

export CUDA_VISIBLE_DEVICES=0
bash ./scripts/scripts/scripts/XFormer/ettm2.sh &

export CUDA_VISIBLE_DEVICES=1
bash ./scripts/scripts/scripts/XFormer/ettm1.sh 

export CUDA_VISIBLE_DEVICES=0
bash ./scripts/scripts/scripts/XFormer/weather.sh &



