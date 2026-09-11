import os 

import numpy as np
import glob 
import os.path as osp
import json

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVAntiUAVDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.uavantiuav_path)
        anno_files = sorted(glob.glob(os.path.join(self.base_path, "*/groundtruth_rect.txt")))

        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]

        self.sequence_list = seq_names


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])
    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = np.loadtxt(anno_path, delimiter=',')
        gt = np.array(ground_truth_rect, dtype = np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith('.jpg') and frame[:-4].isdigit()]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'uavantiuav', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)
