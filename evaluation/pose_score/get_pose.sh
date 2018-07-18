# !/bin/bash

cd /local-scratch/cjc/openpose/

openpose=./build/examples/openpose/openpose.bin

image_dir=$1
resolution=$2
target_dir=$3

mkdir $target_dir

$openpose --image_dir $image_dir --no_display --write_keypoint_json $target_dir --net_resolution $resolution
echo "complete processing $image_dir"


# ./build/examples/openpose/openpose.bin --image_dir  /local-scratch/cjc/ActivityDataset/seq04  --net_resolution 1280x720
