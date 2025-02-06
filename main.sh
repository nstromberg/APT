#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate apt_env

NAME="train-epoch-200-apt-classification"
DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name $NAME"
PYARGS="$PYARGS --artifact_path $DATA/artifacts"
# PYARGS="$PYARGS --data_path $DATA/datasets/regression/OpenML-CTR23"
# PYARGS="$PYARGS --eval_data Moneyball.pt,airfoil_self_noise.pt,concrete_compressive_strength.pt,energy_efficiency.pt,forest_fires.pt,geographical_origin_of_music.pt,solar_flare.pt,student_performance_por.pt,QSAR_fish_toxicity.pt,red_wine.pt,socmob.pt,cars.pt"
PYARGS="$PYARGS --classification"
PYARGS="$PYARGS --data_path $DATA/datasets/classification/OpenML-CC18"
PYARGS="$PYARGS --eval_data mfeat-fourier.pt,breast-w.pt,mfeat-karhunen.pt,mfeat-morphological.pt,mfeat-zernike.pt,cmc.pt,credit-approval.pt,credit-g.pt,diabetes.pt,tic-tac-toe.pt,vehicle.pt,eucalyptus.pt,analcatdata_authorship.pt,pc4.pt,pc3.pt,kc2.pt,blood-transfusion-service-center.pt,cnae-9.pt,ilpd.pt,wdbc.pt,dresses-sales.pt,MiceProtein.pt,steel-plates-fault.pt,climate-model-simulation-crashes.pt,balance-scale.pt,mfeat-factors.pt,vowel.pt,analcatdata_dmft.pt,pc1.pt,banknote-authentication.pt,qsar-biodeg.pt,semeion.pt,cylinder-bands.pt,car.pt,mfeat-pixel.pt"
PYARGS="$PYARGS --device cuda:0"

#PYARGS="$PYARGS --mp"
PYARGS="$PYARGS --max_epochs 200"
PYARGS="$PYARGS --batch_size 64"
PYARGS="$PYARGS --steps_per_epoch 1000"
PYARGS="$PYARGS --aggregate_k_gradients 2"
PYARGS="$PYARGS --num_datasets 16"
PYARGS="$PYARGS --num_trained_datasets 2"
PYARGS="$PYARGS --checkpoint_freq 1"
PYARGS="$PYARGS --reset_freq 2"
PYARGS="$PYARGS --data_lr 0.1"
PYARGS="$PYARGS --initial_eval"

mkdir -p logs/
nohup python main.py $PYARGS > logs/$NAME.out 2>&1 &
