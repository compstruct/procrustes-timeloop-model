terminate=300
threads=32
dataflow=$1
batchsize=$2
array_size=$3
phase=$4
# export TIMELOOP_PROBLEM_SHAPE_DIR=/home/dingqing/timeloop-dev/timeloop/problem-shapes
export TIMELOOP_GLOBAL_SORT=True
# imagenet
python sample.py \
	--phase $phase \
	--batchsize $batchsize \
	--dataset imagenet \
	--net mobilenetv2 \
	--dataflow $dataflow \
	--dense False \
	--synthetic False \
	--sparsity 0.1 \
	--replication False \
  --array_width $array_size \
  --glb_scaling True \
	--terminate $terminate \
	--threads $threads \
	-o 6_14
unset TIMELOOP_GLOBAL_SORT
