import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numbers
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        print(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class Same_Padding(nn.Module):
    def __init__(self, conv_stride=1, conv_ksize=3):
        super(Same_Padding, self).__init__()
        self.conv_stride = conv_stride
        self.conv_ksize = conv_ksize
        
    def forward(self, x):
        B, C, H, W = x.size()
        H_padding = int((H*(self.conv_stride-1)-self.conv_stride+self.conv_ksize)/2)
        W_padding = int((W*(self.conv_stride-1)-self.conv_stride+self.conv_ksize)/2)
        
        return F.pad(x, (W_padding, W_padding, H_padding, H_padding),  mode='replicate')
        
        
gaussian_k = 5       
smoothing = GaussianSmoothing(3, gaussian_k, 1)
same_padding = Same_Padding(conv_stride=1, conv_ksize=gaussian_k)

x = torch.rand(1, 3, 5, 5)
print('original x size: ',x.size())
smoothing_out_nopadding = smoothing(x)
print('smoothing_out_nopadding x size: ',smoothing_out_nopadding.size())
same_padding_out = same_padding(x)
print('same padding out x size: ',same_padding_out.size())
smoothing_out_withpadding = smoothing(same_padding_out)
print('smoothing_out_with padding x size: ',smoothing_out_withpadding.size())

print(x)
print(same_padding_out)


