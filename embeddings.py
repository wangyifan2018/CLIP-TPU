import clip
import os
import time
from tqdm import tqdm
import pickle
import torch
import argparse
from utils import ClipSearchDataset
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)


def compute_embeddings(args):
    if not os.path.exists(os.path.dirname(args.save_path)) and args.mode is None:
        os.makedirs(os.path.dirname(args.save_path))
    model, preprocess = clip.load(args)

    total_start_time = time.time()
    dataset = ClipSearchDataset(img_dir = args.img_dir, preprocess = preprocess)
    dataloader = DataLoader(dataset, batch_size=model.image_net_batch_size, shuffle=False, num_workers=args.num_workers)
    img_path_list, embedding_list = [], []

    for img, img_path in tqdm(dataloader):
        with torch.no_grad():
            features = model.encode_image(img)
            features /= features.norm(dim=-1, keepdim=True)
            embedding_list.extend(features.detach().cpu().numpy())
            img_path_list.extend(img_path)

    result = {'img_path': img_path_list, 'embedding': embedding_list}
    with open(args.save_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)
        logging.info("Results save to {}".format(args.save_path))

    total_image = len(dataset)
    time_use = (time.time()-total_start_time)*1000
    avg_time = time_use/total_image
    logging.info("------------------ Image Encode Time Info ----------------------")
    logging.info("Total images: {}".format(total_image))
    logging.info("Total time use: {:.2f}ms".format(time_use))
    logging.info("Avg time use: {:.2f}ms".format(avg_time))
    logging.info("{:.2f} FPS".format(1000/avg_time))
    logging.info("post_process thread exit!")

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='CLIP.png', help='path of input')
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=4, help='dev id')
    parser.add_argument('--img_dir', type=str, default='./datasets/imagenet_val_1k', help='Directory of images.')
    parser.add_argument('--save_path', type=str, default='./results/embeddings.pkl', help='Path to save the embeddings.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader.')
    parser.add_argument('--update_dir', type=str, default=None, help='if mode is update, need provide update dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    compute_embeddings(args)
    logging.info('all done.')