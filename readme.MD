# CARD

This the Official repository contains PyTorch codes for CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting [paper](https://arxiv.org/abs/2305.12095).

## Designs

**Channel Alignment**: Allow information share among different channels/covariates.
**Dual Attention**: Explore the within patch information.
**Token Blend**: Utilize mutli-scale knowledge.

|![fig1](./pic/main.png)|
|:--:|
|*Figure 1. overall structure of CARD*|
|![alt text](./pic/dual.png)|![alt text](./pic/blend.png)|

## Main Results



## Get Started

1. Dataset can be obtained from Time Series Library (TSlib) at <https://github.com/thuml/Time-Series-Library/tree/main> 

2. The code for long-term forecasting experiment in section 5.1 is in folder `long_term_forecast_l96`. We provide the experiment scripts of all benchmarks under the folder `long_term_forecast_l96/scripts/CARD`. You can reproduce the multivariate experiments by running the following shell scripts:

```
cd long_term_forecast_l96
bash scripts/CARD/ETT.sh 
bash scripts/CARD/wEATHER.sh 
bash scripts/CARD/ECL.sh 
bash scripts/CARD/Traffic.sh 
```

3. The code for long-term forecasting experiment in Appendix E is in folder `long_term_forecast_l720`. We provide the experiment scripts of all benchmarks under the folder `long_term_forecast_l720/scripts/CARD`. You can reproduce the multivariate experiments by running the following shell scripts:

```
cd long_term_forecast_l720
bash scripts/CARD/ettm1.sh
bash scripts/CARD/ettm2.sh
bash scripts/CARD/etth1.sh
bash scripts/CARD/etth2.sh
bash scripts/CARD/weather.sh
bash scripts/CARD/electricity.sh
bash scripts/CARD/traffic.sh
```


4. The code for short-term M4 forecasting experiment in section 5.2 is in folder `short_term_forecast_m4`. We provide the experiment scripts of all benchmarks under the folder `short_term_forecast_m4/scripts/CARD`. You can reproduce the multivariate experiments by running the following shell scripts:

```
cd short_term_forecast_m4
bash scripts/CARD_M4.sh 
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base:

https://github.com/yuqinie98/PatchTST
https://github.com/thuml/Time-Series-Library


## Contact

If you have any questions or concerns, please contact us: xue.w@alibaba-inc.com or tian.zt@alibaba-inc.com


## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{xue2023make,
  title={CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting},
  author={Xue, Wang and Zhou, Tian and Wen, QingSong and Gao, Jinyang and Ding, Bolin and Jin, Rong},
  journal={arXiv preprint arXiv:2305.12095},
  year={2023}
}
```


<!-- ```
@inproceedings{xue2023make,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and 
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
``` -->