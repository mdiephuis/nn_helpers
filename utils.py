import numpy as np
import torch
import torch.nn as nn
import torchvision


# https://github.com/jramapuram/helpers/utils.py
def ones(shape, cuda, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).zero_() + 1


# https://github.com/jramapuram/helpers/utils.py
def zeros(shape, cuda, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).zero_()


# https://github.com/jramapuram/helpers/utils.py
def randn(shape, cuda, mean=0, sigma=1, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).normal_(mean, sigma)


def eye(n_elem, cuda, dtype='float32'):
    return torch.eye(n_elem).type(type_map[dtype](cuda))


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def type_tfloat(use_cuda=False):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def type_tint(use_cuda=False):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def type_tlong(use_cuda=False):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def dummy_context():
    yield None


def to_cuda(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cuda()
    return tensor


def is_cuda(tensor):
    return isinstance(tensor, torch.cuda.Tensor)


def sample_uniform(shape, use_cuda, a=-1, b=1):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_tfloat(use_cuda)(*shape).uniform_(a, b)


def sample_normal(shape, use_cuda, mu=0, sigma=1):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_tfloat(use_cuda)(*shape).normal_(mu, sigma)


def one_hot_np(labels, n_class):
    vec = np.zeros((labels.shape[0], n_class))
    for ind, label in enumerate(labels):
        vec[ind, label] = 1
    return vec


def one_hot(labels, n_class):

    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)

    mask = type_tdouble()(labels.size(0), n_class).fill_(0)

    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)


def hot2label_np(x):
    return np.argmax(x, axis=1)


def hot2label(x):
    return type_tlong()(torch.argmax(x.type(torch.LongTensor), dim=1))


def nan_check_and_break(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check_and_break(tensor, name))
    else:
        if nan_check(tensor, name) is True:
            exit(-1)


def nan_check(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check(tensor, name))
    else:
        if torch.sum(torch.isnan(tensor)) > 0:
            print("Tensor {} with shape {} was NaN.".format(name, tensor.shape))
            return True

        elif torch.sum(torch.isinf(tensor)) > 0:
            print("Tensor {} with shape {} was Inf.".format(name, tensor.shape))
            return True

    return False


def zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() > 0:
        print("tensor {} of {} dim contained ZERO!!".format(name, tensor.shape))
        exit(-1)


def all_zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() == np.prod(list(tensor.shape)):
        print("tensor {} of {} dim was all zero".format(name, tensor.shape))
        exit(-1)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


def pca_project(x, num_elem=2):

    if isinstance(x, torch.Tensor) and len(x.size()) == 3:
        batch_proj = []
        for batch_ind in range(x.size(0)):
            tensor_proj = pca_project(x[batch_ind].squeeze(0), num_elem)
            batch_proj.append(tensor_proj)
        return torch.cat(batch_proj)

    xm = x - torch.mean(x, 1, keepdim=True)
    xx = torch.matmul(xm, torch.transpose(xm, 0, -1))
    u, s, _ = torch.svd(xx)
    x_proj = torch.matmul(u[:, 0:num_elem], torch.diag(s[0:num_elem]))
    return x_proj


def plot_tensor_grid(batch_tensor, save_filename=None):
    ''' Helper to visualize a batch of images.
        A non-None filename saves instead of doing a show()
        This function needs the following to be set, prior to use
            import matplotlib
            matplotlib.use('Agg')
    '''
    import matplotlib.pyplot as plt

    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    if save_filename is not None:
        torchvision.utils.save_image(batch_tensor, save_filename, padding=5)
    else:
        plt.show()


# https://github.com/jramapuram/helpers/utils.py
type_map = {
    'float32': type_tfloat,
    'float64': type_tdouble,
    'double': type_tdouble,
    'int32': type_tint,
    'int64': type_tlong
}
