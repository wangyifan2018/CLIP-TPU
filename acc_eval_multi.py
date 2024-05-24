import os
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision.datasets import CIFAR100
import logging
logging.basicConfig(level=logging.INFO)

from clip import CLIP_Multi

def save_images(images):
    image_names = []
    if not os.path.exists('tmp'):
        os.makedirs('tmp')  # 如果tmp文件夹不存在，则创建
    for i, image in enumerate(images):
        image_path = os.path.join('tmp', f'image_{i}.png')  # 构造图像路径
        image.save(image_path)  # 保存图像
        image_names.append(image_path)  # 将图像路径添加到列表中
    return image_names

def main(args):
    # Load the dataset
    root = os.path.expanduser("./datasets/")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False)


    def get_features(dataset):
        all_features = []
        all_labels = []

        images, labels = zip(*dataset)

        logging.info("prepare test images")
        image_names = save_images(images)
        args.img_dir = image_names
        clip_multi =  CLIP_Multi(args.image_model, args.text_model, args.img_dir, args.save_path, args.max_que_size, args.dev_id)
        features = clip_multi.encode_image()

        all_features.append(features)
        all_labels.append(torch.tensor(labels))

        return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    logging.info(f"Accuracy = {accuracy:.3f}")


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='CLIP.png', help='path of input')
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--max_que_size', type=int, default=128, help='Max size of queue.')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--save_path', type=str, default='./results/embeddings.pkl', help='Path to save the embeddings.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')