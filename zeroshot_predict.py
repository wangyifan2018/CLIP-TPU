import os
from torchvision.datasets import CIFAR100
import torch
import clip
import argparse
import logging
logging.basicConfig(level=logging.INFO)


def main(args):
    # Load the model
    model, preprocess = clip.load(args.image_model, args.text_model, args.dev_id)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("./datasets"), download=True, train=False)

    # Prepare the inputs
    image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes])

    # Calculate features
    with torch.no_grad():
        # print(image_features.shape)
        image_features = model.encode_image(image_input)
        # print(image_features.shape)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    logging.info("------------------- Preprocess Time ------------------------")
    logging.info("Image num {}, proprocess average time (ms): {:.2f}".format(image_input.shape[0], model.preprocess_time / image_input.shape[0] * 1000))

    logging.info("------------------ Text Encoding Time ----------------------")
    logging.info("Image num {}, text encoding average time (ms): {:.2f}".format(image_input.shape[0], model.encode_text_time / image_input.shape[0] * 1000))

    logging.info("------------------ Image Encoding Time ----------------------")
    logging.info("Image num {}, image encoding average time (ms): {:.2f}".format(image_input.shape[0], model.encode_image_time / image_input.shape[0] * 1000))


    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='CLIP.png', help='path of input')
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')