flag=$1
mode=$1

cuda=0
batch_size=5
val_step=10000000
log_step=500

vse_rl_alpha=0.0
vse_mt_alpha=0.0
vse_lm_alpha=1.0
vse_bc_alpha=0.0
vse_h_alpha=0.00

seed=1213

save_path="/home/s1847450/data/vsyntax/"
data_path="/afs/inf.ed.ac.uk/group/project/s1847450/mscoco/"
data_path="/home/s1847450/data/vsyntax/mscoco/"

model_name="v.pcfg"
model_name="v.pcfg.sort"
model_name="v.pcfg.sort.real"
model_name="v.pcfg.sort.real.fix"
model_name="v.test"

model_name="v.rnd.pcfg"
model_name="v.rnd.pcfg.reg"
model_name="v.rnd.pcfg.reg.05"
model_name="v.rnd.pcfg.reg.05.05"
model_name="v.rnd.pcfg.again"
model_name="v.rnd.pcfg.reg.00.05"
model_name="v.rnd.test"

model_name="v.rnd.pcfg."$seed
model_name="v.rnd.pcfg."$seed.copy

log_path=$save_path$model_name
log_file=$log_path/"stdout.txt"

model_init=$save_path"v.rnd.pcfg/-1.pth.tar"
model_init=""

#echo "Be careful this may delete "$this_model
model_tar=$2
model_file=$save_path$model_name/$model_tar".pth.tar"

if [ "$mode" = "test" ]; then
    CUDA_VISIBLE_DEVICES=$cuda python train.py \
        --batch_size $batch_size \
        --data_path $data_path \
        --val_step $val_step \
        --log_step 1 \
        --vse_rl_alpha 0.0 \
        --vse_mt_alpha 0.0 \
        --vse_lm_alpha 1.0 \
        --vse_bc_alpha 1.0 \
        --vse_h_alpha 0.1 \
        --model_init "$model_init" \
        --seed $seed \
        --gpu $cuda \
        --logger_name $log_path 
elif [ "$mode" = "toy" ]; then
    CUDA_VISIBLE_DEVICES=$cuda python train.py \
        --batch_size 2 \
        --word_dim 3 \
        --lstm_dim 2 \
        --embed_size 3 \
        --nt_states 3 \
        --t_states 2 \
        --data_path $data_path \
        --val_step 1 \
        --log_step 1 \
        --logger_name $log_path 
elif [ "$mode" = "eval" ]; then
    data_name='dev' #'test'
    gold_name='val_caps.ground.xx' #'test_ground-truth.txt'
    data_name='val'
    gold_name='test_ground-truth.txt'
    data_name='val'
    data_name='val_gold'
    CUDA_VISIBLE_DEVICES=$cuda python eval.py --candidate $model_file \
        --data_name $data_name \
        --gold_name $gold_name
elif [ "$mode" = "train" ]; then
    echo "running in the background..."
    CUDA_VISIBLE_DEVICES=$cuda nohup python train.py \
        --batch_size $batch_size \
        --data_path $data_path \
        --val_step $val_step \
        --log_step $log_step \
        --vse_rl_alpha $vse_rl_alpha \
        --vse_mt_alpha $vse_mt_alpha \
        --vse_lm_alpha $vse_lm_alpha \
        --vse_bc_alpha $vse_bc_alpha \
        --vse_h_alpha $vse_h_alpha \
        --model_init "$model_init" \
        --seed $seed \
        --gpu $cuda \
        --logger_name $log_path > $log_file 2>&1 &
fi

