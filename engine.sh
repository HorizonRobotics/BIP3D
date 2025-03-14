export PYTHONPATH=./:$PYTHONPATH
CONFIG=configs/bip3d_det.py
if ! [[ -z $1 ]]; then
    echo $1
    CONFIG=$1
fi
nvcc -V
which nvcc

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}
echo $CONFIG

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${CONFIG} \
        ${gpu_num} \
        --work-dir=work-dirs
else
    python ./tools/train.py \
        ${CONFIG} \
        --work-dir ./work-dir
fi
