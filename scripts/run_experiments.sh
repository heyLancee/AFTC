#!/bin/bash

# Script to reproduce results

# 切换到上一级目录
cd ..

# 打印当前目录，确认是否切换成功
echo "当前工作目录: $(pwd)"

# pip
pip install pandas
pip install scikit-learn
pip install gym

# 启动多个训练任务，分别保存到不同的文件夹
for i in {1..5}; do
    current_time=$(date "+%Y-%m-%d_%H-%M-%S")
    dir_name="${current_time}_${i}"

    # 启动训练任务，并将输出保存在不同的文件夹中
    python main.py \
    --dir ${dir_name} \
    --policy "TD3" \
    --env "SunPointFaultSatellite" \
    --seed $i \
    --start_timesteps 4000 \
    --load_model "" \
    --policy_hidden_size 512 \
    --dyn_hidden_size 64 128 \
    --lr 0.0005 \
    --policy_noise 0.1 \
    --noise_clip 0.3 \
    --policy_freq 2 \
    --dyn_net_path "models/dynamic_net/attitude_dynamics_model.pth" \
    --max_timesteps 2000000  &
done

# 等待所有任务完成
wait
