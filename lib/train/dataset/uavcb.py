import csv
import os
import os.path
import random
from collections import OrderedDict

import glob 
import os.path as osp
import json

import numpy as np
import pandas
import torch

from lib.train.admin import env_settings      
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


class UAVCB(BaseVideoDataset):

    def __init__(self, root = None, image_loader = jpeg4py_loader, split = None, seq_ids = None, data_fraction = None):
        root = env_settings().uavantiuav if root is None else root
        super().__init__('UAVCB', root, image_loader)
        print("Dataset root:", self.root)

        anno_files = sorted(glob.glob(os.path.join(self.root, '*/groundtruth_rect.txt'))) 
        print("Found GT", len(anno_files))

        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            num_sequences = int(len(self.sequence_list) * data_fraction)
            self.sequence_list = random.sample(self.sequence_list,  num_sequences)

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'uavcb'

    def has_class_info(self):
        return True
    
    def has_occlusion_info(self):
        return True
    
    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info
    def _read_meta(self, seq_path):
        return OrderedDict({
            'object_class_name': 'UAV',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })
    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]
    def _get_sequence_list(self):
        anno_files = sorted(glob.glob(os.path.join(self.root, '*/groundtruth_rect.txt')))

        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]
        print(len(seq_dirs))
        return seq_names

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')

        ground_truth_rect = np.loadtxt(bb_anno_file, delimiter=',')
        gt = np.array(ground_truth_rect, dtype = np.float64)

        return torch.tensor(gt)

    def _read_target_visible(self, seq_path): 
        gt_file = os.path.join(seq_path, 'groundtruth_rect.txt') # Đọc ground truth để biết số lượng frame 
        gt = np.loadtxt(gt_file, dtype=np.float32, delimiter=',') # Đảm bảo gt luôn có dạng [N, 4] 
        gt = np.asarray(gt).reshape(-1, 4) 
        num_frames = len(gt) 
        target_visible = torch.ones( num_frames, dtype=torch.uint8 ) 
        visible_ratio = target_visible.float() 
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)  
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno = True):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is True:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
