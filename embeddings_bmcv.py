#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
from clip import CLIP_Multi
import logging
logging.basicConfig(level=logging.INFO)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--img_dir', type=str, default='./datasets/imagenet_val_1k', help='Directory of images.')
    parser.add_argument('--save_path', type=str, default='./results/embeddings.pkl', help='Path to save the embeddings.')
    parser.add_argument('--max_que_size', type=int, default=128, help='Max size of queue.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    clip_multi =  CLIP_Multi(args.image_model, args.text_model, args.img_dir, args.save_path, args.max_que_size, args.dev_id)
    features = clip_multi.encode_image()
    logging.info('all done.')
