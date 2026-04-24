#! /bin/bash

#ADME_HLM、ADME_hPPB、ADME_MDR1_ER、ADME_RLM、ADME_rPPB、ADME_Sol

for DATASET_NAME in {"cmax_scaffold_split",} #cmax_random_split, cmax_scaffold_split

do
  for SEED in {1998,2023,2025,2022,2024} #1998,2023,2025,2022,2024
  do
    
    PROJECT_DIRECTORY="/home/datahouse1/liujin/project_simm/MotifAttnNet"


    python ./main.py         --seed ${SEED} \
                              --device "cuda:2" \
                              --config_path "config.json" \
                              --datafolder ${PROJECT_DIRECTORY}/datasets/Cmax/${DATASET_NAME} \
                              --result_path ${PROJECT_DIRECTORY}/result/Cmax/${DATASET_NAME}_run${SEED} \
                              --save_interpret True \
                              --regression_task True \
                              --epochs 200  --dose_mode True\
                              --evaluate_epoch 1 --evaluate_step 500 \
                              --lrate 1e-4  --eps 1e-05  --betas "(0.9,0.999)"  --batch_size 64 \
                              
      done
  done


# nohup bash run_cmax_dose_random.sh >run_cmax_dose_random250202.sh.log 2>&1 &
