import json
import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from data.nsa import patch_ex

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
# mean_train = [0.5, 0.5, 0.5]
# std_train = [0.5, 0.5, 0.5]

def MVTecDataset_cad(type, root, setting=4):

    if setting == 1 or setting == 2:
        task_num = 2
    elif setting == 3:
        task_num = 5
    elif setting == 4:
        task_num = 6
    else:
        assert "setting not exist"

    Dataset_list = []
    for task_id in range(task_num):
        Dataset_list.append(MVTecDataset_task(type, root, setting, task_id + 1))

    return Dataset_list, task_num


class MVTecDataset_task(Dataset):
    def __init__(self,type, root, setting, task_id):
        self.data = []
        self.type = type
        if type == 'train':
            with open(f'./data/MVTec-AD/{setting}/train_{task_id}.json', 'rt') as f:
                self.data = json.load(f)
        else:
            with open(f'./data/MVTec-AD/{setting}/test_{task_id}.json', 'rt') as f:
                self.data = json.load(f)

        self.label_to_idx = {'bottle': '0', 'cable': '1', 'capsule': '2', 'carpet': '3', 'grid': '4', 'hazelnut': '5',
                             'leather': '6', 'metal_nut': '7', 'pill': '8', 'screw': '9', 'tile': '10',
                             'toothbrush': '11', 'transistor': '12', 'wood': '13', 'zipper': '14'}
        self.image_size = (256, 256)
        self.root = root


    def __len__(self):
        return len(self.data)

    def find_idx(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise ValueError("Index out of bounds")

        target_label = self.data[idx]['clsname']

        possible_indices = [i for i in range(len(self.data)) if self.data[i]['clsname'] == target_label and i != idx]

        if not possible_indices:
            return idx

        return random.choice(possible_indices)

    def get_nsa_args(self, clsname):
        return {'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                'num_patches': 4,
                }

    def __getitem__(self, idx):
        item = self.data[idx]

        if idx % 2 == 0 and self.type == 'train':
            nsa_idx = self.find_idx(idx)
            sample_type = 'nsa' if nsa_idx != idx else '_nsa'
        else:
            sample_type = '_nsa'
            nsa_idx = idx


        nsa_item = self.data[nsa_idx]

        source_filename = item['filename']
        nsa_filename = nsa_item['filename']

        transform_fn = transforms.Resize(self.image_size)

        target = cv2.imread(os.path.join(self.root, source_filename))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(target, "RGB")
        target = transform_fn(target)

        source = cv2.imread(os.path.join(self.root, nsa_filename))
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(source, "RGB")
        source = transform_fn(source)

        label = item["label"]
        if sample_type == 'nsa':
            source, mask = patch_ex(np.asarray(target), np.asarray(source), **self.get_nsa_args(item['clsname']))
            mask = (mask[:, :, 0] * 255.0).astype(np.uint8)
        else:
            if item.get("maskname", None):
                mask = cv2.imread(os.path.join(self.root, item['maskname']), cv2.IMREAD_GRAYSCALE)
            else:
                if label == 0:  # good
                    mask = np.zeros(self.image_size).astype(np.uint8)
                elif label == 1:  # defective
                    mask = (np.ones(self.image_size) * 255.0).astype(np.uint8)
                else:
                    raise ValueError("Labels must be [None, 0, 1]!")

        target = transforms.ToTensor()(target)
        source = transforms.ToTensor()(source)

        prompt = ""

        mask = Image.fromarray(mask, "L")
        mask = transform_fn(mask)
        mask = transforms.ToTensor()(mask)

        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["clsname"]
        image_idx = self.label_to_idx[clsname]

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))
