import numpy as np
import torch
import time
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CLIP:
    def __init__(self, args):
        # image bmodel
        self.image_net = sail.Engine(args.image_model, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.image_model))
        self.image_net_graph_name = self.image_net.get_graph_names()[0]
        self.image_net_input_name = self.image_net.get_input_names(self.image_net_graph_name)[0]
        self.image_net_output_name = self.image_net.get_output_names(self.image_net_graph_name)[0]
        self.image_net_input_shape = self.image_net.get_input_shape(self.image_net_graph_name, self.image_net_input_name)
        self.image_net_output_shape = self.image_net.get_output_shape(self.image_net_graph_name, self.image_net_output_name)
        self.image_net_batch_size = self.image_net_input_shape[0]

        self.image_resolution = self.image_net_input_shape[2] # 224 for vit32-b
        self.embed_dim = self.image_net_output_shape[1] # 512 for vit32-b

        # text bmodel
        self.text_net = sail.Engine(args.text_model, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.text_model))
        self.text_net_graph_name = self.text_net.get_graph_names()[0]
        self.text_net_input_name = self.text_net.get_input_names(self.text_net_graph_name)[0]
        self.text_net_output_name = self.text_net.get_output_names(self.text_net_graph_name)[0]
        self.text_net_input_shape = self.text_net.get_input_shape(self.text_net_graph_name, self.text_net_input_name)
        self.text_net_batch_size = self.text_net_input_shape[0]

        self.text_projection = torch.from_numpy(np.load('./models/BM1684X/text_projection_512_512.npy'))
        self.logit_scale = torch.tensor(4.605170249938965) # nn.param from torch model

        # init preprocess
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.preprocess_cpu = Compose([
            Resize(self.image_resolution, interpolation=BICUBIC),
            CenterCrop(self.image_resolution),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(self.mean, self.std),
        ])

        self.encode_image_time = 0.0
        self.encode_text_time = 0.0
        self.preprocess_time = 0.0

    def preprocess(self, image):
        start_time = time.time()
        image = self.preprocess_cpu(image)
        self.preprocess_time += time.time() - start_time
        return image


    def encode_image(self, image: torch.tensor):
        start_time = time.time()
        image_batch = image.shape[0]
        processed_outputs = []
        if image_batch > self.image_net_batch_size:
            for start_idx in range(0, image_batch, self.image_net_batch_size):
                end_idx = min(start_idx + self.image_net_batch_size, image_batch)  # Ensure end_idx does not exceed image_batch
                batch_slice = image[start_idx:end_idx]
                if batch_slice.shape[0] < self.image_net_batch_size:
                    padding_size = self.image_net_batch_size - batch_slice.shape[0]
                    batch_slice = torch.cat([batch_slice, torch.zeros(padding_size, *batch_slice.shape[1:])], dim=0)
                input_data = {self.image_net_input_name: batch_slice}
                results = torch.from_numpy(self.image_net.process(self.image_net_graph_name, input_data)[self.image_net_output_name])
                processed_outputs.append(results)
        else:
            padding_image = None
            if image_batch < self.image_net_batch_size:
                padding_image = torch.cat([image, torch.zeros(self.image_net_batch_size - image_batch, *image.shape[1:])], dim=0)
            else:
                padding_image = image
            input_data = {self.image_net_input_name: padding_image}
            results = torch.from_numpy(self.image_net.process(self.image_net_graph_name, input_data)[self.image_net_output_name])
            processed_outputs.append(results)

        processed_outputs = torch.cat(processed_outputs, dim=0)
        self.encode_image_time += time.time() - start_time
        return processed_outputs[:image_batch]  # Trim padding off the final output if it was padded


    def encode_text(self, text):
        start_time = time.time()
        text_batch = text.shape[0]
        processed_outputs = []
        if text_batch > self.text_net_batch_size:
            for start_idx in range(0, text_batch, self.text_net_batch_size):
                end_idx = min(start_idx + self.text_net_batch_size, text_batch)  # Ensure end_idx does not exceed text_batch
                batch_slice = text[start_idx:end_idx]
                if batch_slice.shape[0] < self.text_net_batch_size:
                    padding_size = self.text_net_batch_size - batch_slice.shape[0]
                    batch_slice = torch.cat([batch_slice, torch.zeros(padding_size, *batch_slice.shape[1:])], dim=0)
                input_data = {self.text_net_input_name: batch_slice}
                results = torch.from_numpy(self.text_net.process(self.text_net_graph_name, input_data)[self.text_net_output_name])
                processed_outputs.append(results)
        else:
            padding_text = None
            if text_batch < self.text_net_batch_size:
                padding_size = self.text_net_batch_size - text_batch
                padding_text = torch.cat([text, torch.zeros(padding_size, *text.shape[1:])], dim=0)
            else:
                padding_text = text
            input_data = {self.text_net_input_name: padding_text}
            results = torch.from_numpy(self.text_net.process(self.text_net_graph_name, input_data)[self.text_net_output_name])
            processed_outputs.append(results)

        processed_outputs = torch.cat(processed_outputs, dim=0)[:text_batch] # Trim padding off the final output if it was padded
        processed_outputs = processed_outputs[torch.arange(processed_outputs.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        self.encode_text_time += time.time() - start_time
        return processed_outputs


    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
