from PIL import Image

import numpy as np
import torch

# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
class VOC2012Dataset(torch.utils.data.Dataset):

    classes = np.array(['background','aeroplane','bicycle','bird','boat','bottle','bus','car',
                        'cat','chair','cow','diningtable','dog','horse','motorbike','person',
                        'potted plant','sheep','sofa','train','tv/monitor'])

    ignore_index = 255

    def __init__(self, root, split):

        self.root = root
        self.indices = self.read_indices(split)

    def read_indices(self, split):
        indices_file =  f'{self.root}/ImageSets/Segmentation/{split}.txt'
        with open(indices_file) as f:
            indices = f.read().splitlines()
        return indices

    def read_image(self, index):
        image = Image.open(f'{self.root}/JPEGImages/{index}.jpg')
        return image

    def read_label(self, index):
        label = Image.open(f'{self.root}/SegmentationClass/{index}.png')
        return label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        image = self.read_image(self.indices[index])
        image = np.array(image)
        # channel last -> channel first
        image = image.transpose(2, 0, 1)

        label = self.read_label(self.indices[index])
        label = np.array(label)

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()