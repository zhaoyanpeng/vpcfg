# Visually Grounded Compound PCFGs
Code for the paper *[Visually Grounded Compound PCFGs](https://arxiv.org/abs/2009.12404)* by [Yanpeng Zhao](https://zhaoyanpeng.cn/) and [Ivan Titov](http://ivan-titov.org/) @ EMNLP 2020.

## Data
The processed data can be downloaded [here](https://drive.google.com/file/d/1_igQa9KD8J5CsvtuYsdDNA0QaNTZ4LXD/view?usp=sharing). 

## Learning 
### C-PCFGs on Text
```shell
python train.py \
    --data_path $DATA_PATH \
    --logger_name $SAVE_PATH
```
### C-PCFGs on Image & Text
```shell
python train.py \
    --visual_mode "O" \
    --data_path $DATA_PATH \
    --logger_name $SAVE_PATH
```
In both cases please specify `DATA_PATH` and `SAVE_PATH` before running (`DATA_PATH` is where the downloaded data resides; `SAVE_PATH` is where your model will be saved).

## Parsing 
Remember to specify `MODEL_FILE` and `DATA_PATH` first.
```shell
python eval.py \
    --model $MODEL_FILE \
    --data_path $DATA_PATH
```

## Dependencies
It requires a tailored [Torch-Struct](https://github.com/zhaoyanpeng/pytorch-struct).
```shell
git clone --branch beta https://github.com/zhaoyanpeng/vpcfg.git
cd vpcfg
virtualenv -p python3.7 ./pyenv/oops
source ./pyenv/oops/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
cd pytorch-struct
pip install -e .
```

## Acknowledgements
This repo is developed based on [VGNSL](https://github.com/ExplorerFreda/VGNSL), [C-PCFGs](https://github.com/harvardnlp/compound-pcfg), and [Torch-Struct](https://github.com/harvardnlp/pytorch-struct).

## License
MIT
