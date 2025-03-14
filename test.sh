export PYTHONPATH=./:$PYTHONPATH
CONFIG=$1
CKPT=$2

nvcc -V
which nvcc

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}
echo "ckeckpoint: "$CKPT
echo $CONFIG


if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_test.sh \
        ${CONFIG} \
        $CKPT \
        $gpu_num \
        --work-dir ./work-dir
else
    python3 tools/test.py \
        $CONFIG \
        $CKPT \
        --work-dir ./work-dir
fi
