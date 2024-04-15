#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
from tqdm import tqdm
import pickle
import torch
import numpy as np
import argparse
import sophon.sail as sail
import threading
import queue
from utils import get_imagenames
import logging
logging.basicConfig(level=logging.INFO)


class ImgEncodeThread(object):
    def __init__(self, args):
        self.dev_id = args.dev_id
        self.bmodel_name = args.image_model
        self.image_name_list = get_imagenames(args.img_dir)
        self.save_path = args.save_path
        self.max_que_size = args.max_que_size
        self.img_path_list = []
        self.embedding_list = []
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.alpha_beta = [((1.0 / 255) / self.std[i], -self.mean[i] / self.std[i]) for i in range(3)]

        # init sail engine
        self.engine_image_pre_process = sail.EngineImagePreProcess(self.bmodel_name, self.dev_id, False)
        self.engine_image_pre_process.InitImagePreProcess(sail.sail_resize_type.BM_RESIZE_TPU_LINEAR, True, self.max_que_size, self.max_que_size)
        # self.engine_image_pre_process.SetPaddingAtrr(114,114,114,1)
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        self.net_w = self.engine_image_pre_process.get_input_width()
        self.net_h = self.engine_image_pre_process.get_input_height()
        self.output_name = self.engine_image_pre_process.get_output_names()[0]
        self.output_shape = self.engine_image_pre_process.get_output_shape(self.output_name)
        self.batch_size = self.output_shape[0]
        if(len(self.image_name_list)%self.batch_size != 0):
            sub_num = self.batch_size - len(self.image_name_list)%self.batch_size
            for _ in range(sub_num):
                self.image_name_list.append(self.image_name_list[0])

        self.run_count = int(len(self.image_name_list)/self.batch_size)
        self.loop_count = len(self.image_name_list)
        self.post_queue = queue.Queue(self.max_que_size)

    def StartProcess(self):
        thread_decoder = threading.Thread(target=self.decoder_and_pushdata)
        thread_inference = threading.Thread(target=self.Inferences_thread)
        thread_postprocess = threading.Thread(target=self.post_process)

        thread_decoder.start()
        thread_inference.start()
        thread_postprocess.start()

        thread_decoder.join()
        thread_inference.join()
        thread_postprocess.join()

        return torch.tensor(self.embedding_list[:len(self.image_name_list)]).view(len(self.image_name_list), self.output_shape[1])


    def center_crop_bmcv(self, bmcv, bmimg):
        """
        Center crop the bmimg to match the aspect ratio of net_h and net_w.
        :param bmimg: Input BMImage to be cropped.
        :return: Cropped BMImage.
        """
        img_w, img_h = bmimg.width(), bmimg.height()

        # Calculate the target aspect ratio
        target_aspect_ratio = self.net_w / self.net_h
        # Calculate the aspect ratio of the input image
        img_aspect_ratio = img_w / img_h
        if img_aspect_ratio > target_aspect_ratio:
            # If the image is wider than the target aspect ratio, calculate the new width to match the target aspect ratio
            crop_w = int(img_h * target_aspect_ratio)
            crop_h = img_h
        else:
            # If the image is taller than the target aspect ratio, calculate the new height to match the target aspect ratio
            crop_w = img_w
            crop_h = int(img_w / target_aspect_ratio)

        # Calculate the top-left corner of the crop area to ensure center cropping
        crop_x0 = (img_w - crop_w) // 2
        crop_y0 = (img_h - crop_h) // 2

        # Perform the crop using the bmcv.crop method
        cropped_img = bmcv.crop(bmimg, crop_x0, crop_y0, crop_w, crop_h)

        return cropped_img

    def decoder_and_pushdata(self):
        time_start = time.time()
        handle = sail.Handle(self.dev_id)
        bmcv = sail.Bmcv(handle)
        for image_index, image_name in enumerate(self.image_name_list):
            # logging.info(image_name)
            decoder = sail.Decoder(image_name,True,self.dev_id)
            bmimg = decoder.read(handle)
            cropped_img = sail.BMImage()
            cropped_img = self.center_crop_bmcv(bmcv, bmimg)
            while(self.engine_image_pre_process.PushImage(0,image_index, cropped_img) != 0):
                logging.info("engine_image_pre_process.full()")
                time.sleep(0.001)
        using_time = time.time()-time_start
        logging.info("decoder_and_pushdata thread exit, time use: {:.2f}s,avg: {:.2f}ms".format(
            using_time,using_time/len(self.image_name_list)*1000))

    def Inferences_thread(self):
        for _ in range(self.run_count):
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData_Npy()

            while self.post_queue.full():
                time.sleep(0.001)
                logging.info("post_queue.full()")
                continue
            self.post_queue.put([output_tensor_map,imageidx_list],False)
            end_time = time.time()
            # logging.info("GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        logging.info("Inferences_thread thread exit!")


    def post_process(self):
        start_time = time.time()
        for _ in tqdm(range(self.run_count), total=self.run_count ,desc="Image Encode Progress"):
            while self.post_queue.empty():
                time.sleep(0.001)
            output_tensor_map, imageidxs = self.post_queue.get(True)
            features = output_tensor_map[self.output_name]
            features /= np.linalg.norm(features, axis=-1, keepdims=True)
            self.embedding_list.extend(features)
            self.img_path_list.extend([self.image_name_list[idx] for idx in imageidxs])


        end_time = time.time()
        time_use = (end_time-start_time)*1000
        avg_time = time_use/self.loop_count
        logging.info("------------------ Image Encode Time Info ----------------------")
        logging.info("Total images: {}".format(self.loop_count))
        logging.info("Total time use: {:.2f}ms".format(time_use))
        logging.info("Avg time use: {:.2f}ms".format(avg_time))
        logging.info("{:.2f} FPS".format(1000/avg_time))
        logging.info("post_process thread exit!")

        result = {'img_path': self.img_path_list, 'embedding': self.embedding_list}
        with open(self.save_path, 'wb') as f:
            pickle.dump(result, f, protocol=4)
            logging.info("Results save to {}".format(args.save_path))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_32b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--img_dir', type=str, default='./datasets/imagenet_val_1k', help='Directory of images.')
    parser.add_argument('--save_path', type=str, default='./results/embeddings.pkl', help='Path to save the embeddings.')
    parser.add_argument('--max_que_size', type=int, default=128, help='Max size of queue.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    process =  ImgEncodeThread(args)
    features = process.StartProcess()
    logging.info('all done.')