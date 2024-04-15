#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import clip
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)


def main(args):
    # Load the model
    model, preprocess = clip.load(args)
    # Load the dataset
    root = os.path.expanduser("./datasets/")
    train = CIFAR100(root, download=True, train=True, transform=preprocess)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=model.image_net_batch_size)):
                features = model.encode_image(images)

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

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
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_32b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')