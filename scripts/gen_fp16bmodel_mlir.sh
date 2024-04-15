#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "bm1684 do not support fp16"
        exit
    fi
fi

outdir=./$target_dir
function gen_mlir()
{
    model_transform.py \
      --model_name clip_image_vitb32 \
      --model_def ../models/onnx/clip_image_vitb32.onnx \
      --input_shapes [[$1,3,224,224]] \
      --pixel_format rgb \
      --mlir clip_image_vitb32_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir clip_image_vitb32_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model ./clip_image_vitb32_${target}_f16_$1b.bmodel

    mv ./clip_image_vitb32_${target}_f16_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

# batch_size=8
gen_mlir 8
gen_fp16bmodel 8

# batch_size=16
gen_mlir 16
gen_fp16bmodel 16

# batch_size=32
gen_mlir 32
gen_fp16bmodel 32
popd