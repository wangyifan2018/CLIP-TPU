#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

outdir=./$target_dir

function gen_mlir()
{
    model_transform.py \
      --model_name clip_image_vitb32 \
      --model_def ./clip_image_vitb32.onnx \
      --input_shapes [[$1,3,224,224]] \
      --mean 123.675,116.28,103.53 \
      --scale 0.0171,0.0175,0.0174 \
      --pixel_format rgb \
      --mlir clip_image_vitb32_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir clip_image_vitb32_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model ./clip_image_vitb32_${target}_f32_$1b.bmodel

    mv ./clip_image_vitb32_${target}_f32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd