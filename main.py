'''
Created on Jan 19, 2021

@author: Mahbaneh Eshaghzadeh Torbati (mae82@pitt.edu)
'''

from loader_multimodal import Data
from runner import Experiment
import random
from numpy.random import seed
import argparse
from distutils.util import strtobool


#------------------------------ Setting seeds-------------------------------
random.seed(1234)
seed(1234)
#---------------------------------------------------------------------------

def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='The directory of the input folders contain the scans of scanners.')
    
    parser.add_argument('--mask_adr', type=str, default='Data/cropped_JHU_MNI_SS_T1_Brain_Mask.nii',
                        help='The directory of the template brain mask.')
    
    parser.add_argument('--output_dir', type=str, default='Data/Output',
                        help='The directory of the outputs.')
    
    parser.add_argument('--downsample', type=lambda x:bool(strtobool(x)), default=False,
                        help='Whether to downsample input images.')
    
    parser.add_argument('--normalizing', type=lambda x:bool(strtobool(x)), default=False,
                        help='Whether to scale (normalize) images.')
    
    parser.add_argument('--upsampling', type=lambda x:bool(strtobool(x)), default=False,
                        help='Whether to upsample images.')
    
    parser.add_argument('--Swap_axis', type=lambda x:bool(strtobool(x)), default=True,
                        help='Whether to swap axis of images.')
    
    parser.add_argument('--spatial_transformer', type=lambda x:bool(strtobool(x)), default=True,
                        help='Should be always True.')
    
    parser.add_argument('--ind_outs', type=lambda x:bool(strtobool(x)), default=True,
                        help='Should be always True.')
    
    parser.add_argument('--fuse_outs', type=lambda x:bool(strtobool(x)), default=True,
                        help='Should be always True.')
    
    parser.add_argument('--downsample1', type=int, default=13,
                        help='Dimention1 for downsampling images.')
    
    parser.add_argument('--downsample2', type=int, default=10,
                        help='Dimention2 for downsampling images.')
    
    parser.add_argument('--latent_dim', type=int, default=6,
                        help='Number of components (size) of latent embedding (Z).')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for training.')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimizer.')
    
    parser.add_argument('--T1', type=int, default=100,
                        help='Number of epoches for Step1.')
    
    parser.add_argument('--T2', type=int, default=100,
                        help='Number of epoches for Step2.')
    
    
    parser.add_argument('--lambda1', type=float, default = 1.0,
                        help='Loss weight.')
    
    parser.add_argument('--lambda2', type=float, default = 0.3,
                        help='Loss weight.')
    
    parser.add_argument('--lambda3', type=float, default = 1.0,
                        help='Loss weight.')
    
    parser.add_argument('--lambda4', type=float, default = 4.0,
                        help='Loss weight.')
    
    scanner_names = ['ge','philips', 'trio', 'prisma']
    parser.add_argument("--scanner_names", default=scanner_names, 
                        type=lambda s: [item for item in s.split(',')],
                        help = "List of scanners'nmaes.")

    opt = parser.parse_args()
    return opt
 
def main():
    
    #------------------------------ 1.parsing input arguments ------------------
    opt = parse_option()
    #-------------------------------------------------------------------------

    #------------------------------ 2.Loading Data ----------------------------
    # 1.3. Loading data
    data = Data(opt.data_dir, opt.scanner_names, opt.downsample, opt.normalizing,
            opt.downsample1, opt.downsample2, opt.Swap_axis, opt.upsampling, opt.mask_adr)
    data.load()
    #-------------------------------------------------------------------------

    #------------------------------ 3.Pre-processing Data ----------------------
    data.preprocessing()
    #-------------------------------------------------------------------------
    
    # -----------4. Dimension of data after pre-processing for model---------
    W = data.ge[0].shape[1] 
    H = data.ge[0].shape[2]
    #------------------------------------------------------------------------- 
    
    #---------------------------- 5. Running the model ----------------------
    loss_weights = {'lambda_1': opt.lambda1, 'lambda_2': opt.lambda2, 
                'lambda_3': opt.lambda3, 'lambda_4':opt.lambda4}
    print(loss_weights)
    
    exp = Experiment(opt.scanner_names, data, opt.latent_dim, opt.spatial_transformer,
                  opt.ind_outs, opt.fuse_outs, loss_weights, opt.batch_size, opt.T1, 
                  opt.T2, W, H, data.mask,
                  data.mask_affine, opt.learning_rate, opt.output_dir)
    exp.run()
    #--------------------------------------------------------------------------

if __name__ == '__main__':
    main()


