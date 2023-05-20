# CARD

1. Dataset can be obtained from Time Series Library (TSlib) at <https://github.com/thuml/Time-Series-Library/tree/main>

2. The code for long-term forecasting experiment in section 5.1 is in folder `long_term_forecast_l96`. We provide the experiment scripts of all benchmarks under the folder `long_term_forecast_l96/scripts/CARD`. You can reproduce the multivariate experiments by running the following shell scripts:

```
cd long_term_forecast_l96
bash scripts/CARD/ETT.sh 
bash scripts/CARD/wEATHER.sh 
bash scripts/CARD/ECL.sh 
bash scripts/CARD/Traffic.sh 
```


3. The code for short-term M4 forecasting experiment in section 5.2 is in folder `short_term_forecast_m4`. We provide the experiment scripts of all benchmarks under the folder `short_term_forecast_m4/scripts/CARD`. You can reproduce the multivariate experiments by running the following shell scripts:

```
cd short_term_forecast_m4
bash scripts/CARD_M4.sh 
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