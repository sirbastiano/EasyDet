import torch
import torch.nn as nn    
from BBlocks.basics import ResidualDeconvolutionUpsample2d        
                
def layer_upsample(upsample_type):
    upsample_block = {
            "nearest": nn.UpsamplingNearest2d,
            "bilinear": nn.UpsamplingBilinear2d,
            "rdtsc": ResidualDeconvolutionUpsample2d,
            "shuffle": nn.PixelShuffle,
        }[upsample_type]
    return upsample_block


if  __name__    ==  "__main__":
    # Test
    print("Test UpLayers.py")