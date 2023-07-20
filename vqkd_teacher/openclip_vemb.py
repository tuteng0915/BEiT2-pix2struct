# -*- encoding: utf-8 -*-
'''
@File    :   openclip_vemb.py
@Time    :   2023/06/27 19:00:54
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import os
import sys
import math
import random
import torch

from vqkd_teacher.open_clip.factory import create_model_and_transforms
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import normalize
import torch.nn.functional as F


class BaseVemb:
    def __init__(self) -> None:
        pass

    def dense_emb(self, x):
        ''' patch sequence to feature sequence.
        '''
        raise NotImplementedError
    
    def token_emb(self, x):
        ''' patch sequence to token sequence.
        '''
        raise NotImplementedError


class Pix2StructOpenClipVemb():
    def __init__(self, config_path="./vqkd_teacher/open_clip_config_eval.json") -> None:
        super().__init__()
        # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(name)
        model, preprocess_train, preprocess_val = create_model_and_transforms('CLIP-ViT-L-14-DataComp.XL-s13B-b90K', customized_config=config_path, cache_dir='/zhangpai21/checkpoints/clip')
        model.visual.output_token = True
        assert isinstance(preprocess_val, Compose)
        self.mean, self.std = None, None
        for t in preprocess_val.transforms:
            if isinstance(t, Normalize):
                self.mean = torch.Tensor(t.mean)
                self.std = torch.Tensor(t.std)

        assert self.mean is not None
        self.model = model
        self.model.to('cuda')


    def dense_emb(self, x, position_ids, image_size, format="patch"):
        # apply normalize to patch sequence
        if format == 'patch' or format == 'nhwc':
            x = ((x.view(-1, 3) - self.mean) / self.std).view(x.shape)
        elif format == 'nchw':
            x = normalize(x, self.mean, self.std)
        elif format == 'normed_patch':
            x = x
        else:
            raise ValueError(f'Unknown format {format}')
        # apply model
        tokens = self.model.encode_image(x, position_ids=position_ids, image_size=image_size)[:, 1: , :] # TODO return_features is a name to change
        # normalize
        # y = F.normalize(y, dim=-1)
        tokens = F.normalize(tokens, dim=-1)
        return tokens

if __name__ == "__main__":
    Vemb = Pix2StructOpenClipVemb()
    image = torch.randn(1, 400, 14*14*3)
    position_ids = torch.cat([torch.arange(400, dtype=int).unsqueeze(0).unsqueeze(2), torch.zeros(400, dtype=int).unsqueeze(0).unsqueeze(2) ], dim=-1)
    print(position_ids.shape) 
    image_size = (400, 400)
    output = Vemb.dense_emb(image, position_ids, image_size)
    print(output.shape)
