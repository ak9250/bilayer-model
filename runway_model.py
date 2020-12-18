import runway
import numpy as np
from PIL import Image
from infer import InferenceWrapper
import argparse
import torch
from torchvision import transforms
import os.path

args_dict = {
    'project_dir': '.',
    'init_experiment_dir': './runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_dir': './runs/vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

@runway.setup(options={'checkpoint_dir': runway.directory, 'checkpoint_2': runway.directory,})
def setup(opts):
    args_dict['init_experiment_dir'] = os.path.join(opts['checkpoint_dir'], 'vc2-hq_adrianb_paper_main')
    args_dict['experiment_dir'] = os.path.join(opts['checkpoint_dir'], 'vc2-hq_adrianb_paper_enhancer')
    module = InferenceWrapper(args_dict)
    return module

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))

@runway.command('translate', inputs={'source_imgs': runway.image, "target_imgs": runway.image}, outputs={'image': runway.image})
def translate(module, inputs):

    data_dict = {
    'source_imgs': np.array(inputs['source_imgs']), # Size: H x W x 3, type: NumPy RGB uint8 image
    'target_imgs': np.array(inputs['target_imgs']), # Size: NUM_FRAMES x H x W x 3, type: NumPy RGB uint8 images
    }
    data_dict = module(data_dict)
    imgs = data_dict['pred_enh_target_imgs']
    segs = data_dict['pred_target_segs']
    pred_img = to_image(data_dict['pred_enh_target_imgs'][0, 0], data_dict['pred_target_segs'][0, 0])
    return pred_img

if __name__ == '__main__':
    runway.run(port=8889)
    