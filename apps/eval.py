import sys
import os

from numpy.lib.function_base import select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()


class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.load_size, self.load_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def _yaw_extrinsic(self, image_path):
        # Filenames follow <yaw>_0_00.* in this repo; if parsing fails, assume front view (0 deg).
        basename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            yaw_deg = float(basename.split('_')[0])
        except Exception:
            yaw_deg = 0.0

        rad = np.deg2rad(yaw_deg)
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ])

    def load_image(self, images, masks):
        # Name
        img_name = os.path.splitext(os.path.basename(images[0]))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1

        calibList = []
        for image_path in images:
            extrinsic = self._yaw_extrinsic(image_path)
            calib = np.matmul(projection_matrix, extrinsic)
            calibList.append(torch.Tensor(calib).float())

        # Mask
        maskList = []
        imageList = []
        for mask, image in zip(masks, images):
            mask = Image.open(mask).convert('L')
            mask = transforms.Resize((self.load_size, self.load_size))(mask)
            mask = transforms.ToTensor()(mask).float()
            maskList.append(mask)
            image = Image.open(image).convert('RGB')
            image = self.to_tensor(image)
            image = mask.expand_as(image) * image
            imageList.append(image)
        return {
            'name': img_name,
            'img': torch.stack(imageList, dim=0),
            'calib': torch.stack(calibList, dim=0),
            'mask': torch.stack(maskList, dim=0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


def _angle_from_name(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(basename.split('_')[0])
    except Exception:
        return 1e9


def _collect_test_pairs(test_folder_path):
    img_exts = ('*.jpg', '*.jpeg', '*.png')
    images = []
    for ext in img_exts:
        images.extend(glob.glob(os.path.join(test_folder_path, ext)))
    images = [p for p in images if '_mask' not in os.path.basename(p)]
    images = sorted(images, key=lambda p: (_angle_from_name(p), p))

    test_images = []
    test_masks = []
    for image_path in images:
        base, _ = os.path.splitext(image_path)
        mask_candidates = [
            base + '_mask.png',
            base + '.png',
        ]
        mask_path = next((m for m in mask_candidates if os.path.exists(m)), None)
        if mask_path is not None:
            test_images.append(image_path)
            test_masks.append(mask_path)

    return test_images, test_masks


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images, test_masks = _collect_test_pairs(opt.test_folder_path)
    if opt.num_views > 0:
        test_images = test_images[:opt.num_views]
        test_masks = test_masks[:opt.num_views]

    print("Use view:", opt.num_views)
    print("Found %d valid image/mask pairs" % len(test_images))

    if len(test_images) != opt.num_views:
        raise RuntimeError(
            'Expected %d image/mask pairs in %s but found %d. Ensure each image has a matching mask.' %
            (opt.num_views, opt.test_folder_path, len(test_images))
        )

    try:
        data = evaluator.load_image(test_images, test_masks)
        evaluator.eval(data, True)
    except Exception as e:
        print("error:", e.args)
