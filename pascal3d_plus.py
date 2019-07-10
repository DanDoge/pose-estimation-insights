import os
from scipy.io import loadmat
from PIL import Image
import numpy as np
import torch
import pickle
import torch.utils.data as data

objs = pickle.load(open("./data/objs_pascal.pkl", "rb"))

cls = ('__background__', # always index 0
        'aeroplane', 'bicycle', 'boat',
        'bottle', 'bus', 'car', 'chair',
        'diningtable',
        'motorbike',
        'sofa', 'train', 'tvmonitor')

class_to_ind = dict(zip(cls, range(len(cls))))

class PasDet(data.Dataset):
    def __init__(self, pickle_file=objs, transform=None, target_transform=None):
        self.objs = pickle_file
        self.ids = list(tuple(self.objs))
        self.transform = transform
    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open('./data/pascal3d+/Images/' + self.objs[img_id][0]['label'] + '/' + img_id + '.jpg').convert('RGB')
        bboxes = [self.objs[img_id][0]['bbox']]
        viewpoint = [self.objs[img_id][0]['viewpoint']]
        labels = [class_to_ind[self.objs[img_id][0]['label'].split('_')[0]]]
        for obj in self.objs[img_id][1:]:
            bboxes = np.append(bboxes, obj['bbox'])
            viewpoint = np.append(viewpoint, obj['viewpoint'])
            labels = np.append(labels, class_to_ind[obj['label'].split('_')[0]])
        bboxes = np.reshape(bboxes, [-1, 4]).astype(np.int64)
        if self.transform is not None:
            img = self.transform(img)
        return img, {'boxes':torch.LongTensor(bboxes), 'gt_classes': labels, 'viewpoints':viewpoint + labels * 24, 'im_info':list(np.append(img.size, 1))}, img_id
    def __len__(self):
        return 100
        return len(self.ids)

if __name__ == "__main__":
    ds = PasDet(objs)
    print(len(ds))
    print(ds[0])
