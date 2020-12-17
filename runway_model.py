import runway
import numpy as np
from PIL import Image
from infer import InferenceWrapper
import argparse
import torch


args_dict = {
    'project_dir': '.',
    'init_experiment_dir': './runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

@runway.setup
def setup():
    module = InferenceWrapper(args_dict)
    return module

@runway.command('translate', inputs={'source_imgs': runway.image, "target_imgs": runway.image}, outputs={'image': runway.image})
def translate(module, inputs):

    data_dict = {
    'source_imgs': np.array(inputs['source_imgs']), # Size: H x W x 3, type: NumPy RGB uint8 image
    'target_imgs': np.array(inputs['target_imgs']), # Size: NUM_FRAMES x H x W x 3, type: NumPy RGB uint8 images
    }
    data_dict = module(data_dict)
    imgs = data_dict['pred_enh_target_imgs']
    segs = data_dict['pred_target_segs']
    # random_array = np.roll(imgs.cpu(), -1, 1) * 255
    # random_array = random_array.astype(np.uint8)
    # random_image = Image.fromarray(random_array)
    img = Image.fromarray((imgs.cpu().detach().numpy() * 255).astype(np.uint8))
    return img
    
if __name__ == '__main__':
    runway.run(port=8889)