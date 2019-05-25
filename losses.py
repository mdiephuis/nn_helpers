import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as D
from nn_helpers.utils import sample_normal, type_tfloat
from nn_helpers.utils import nan_check_and_break, nan_check, zero_check_and_break


def loss_bce(x, x_hat):
    BCE = F.binary_cross_entropy(
        x_hat.view(-1, 1), x.view(-1, 1), reduction='sum')
    return BCE


def loss_bce_kld(x, x_hat, mu, log_var):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    BCE = F.binary_cross_entropy(
        x_hat.view(-1, 1), x.view(-1, 1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return KLD + BCE


def MSE_kernel(p, q):
    p_tiled = p.expand(p.size(0), q.size(0), p.size(1))
    q_tiled = q.expand(p.size(0), q.size(0), p.size(1)).transpose(0, 1)

    loss_func = torch.nn.MSELoss(reduction='mean')
    nom = - loss_func(p_tiled, q_tiled)

    denom = p.size(1)

    return torch.exp(torch.div(nom, denom))


def max_mean_discrepancy(p, q, kernel_func):
    '''
    Maximum Mean Discrepancy based loss using kernel embedding trick
    Two distributions are identical iff all their moments are the same. (Gretton, 2007)
    '''
    pp = kernel_func(p, p)
    qq = kernel_func(q, q)
    pq = kernel_func(p, q)

    return torch.mean(pp + qq - 2 * pq)


def loss_mmd(x, x_hat, z, use_cuda):
    '''
    Maximizing Variational Autoencoders (MMD-VAE) loss
    '''
    true_samples = sample_normal(z.size(), use_cuda)
    mmd = max_mean_discrepancy(true_samples, z, MSE_kernel)

    return mmd


def loss_elbo(z_mu, z_std):
    elbo = torch.mean(0.5 * torch.sum(torch.pow(z_std, 2) + torch.pow(z_mu, 2) - torch.log(torch.pow(z_std, 2)), dim=1, keepdim=False))
    return elbo


def kl_div_gaussian(z_mu, z_std):
    return torch.mean(0.5 * torch.sum(torch.pow(z_std, 2) + torch.pow(z_mu, 2) - torch.log(torch.pow(z_std, 2)) - 1, dim=1), dim=0)


def conditional_entropy(z_std):
    hxy = torch.mean(torch.sum(torch.log(z_std), dim=1, keepdim=False))
    return hxy


def loss_mse(x, x_hat):
    # Add variance?
    mse_func = torch.nn.MSELoss(reduction='sum')
    mse = mse_func(x, x_hat)
    return mse


def loss_infovae(x, x_hat, z_mu, z_std, alpha, beta, use_cuda, gamma=1.0):
    mse = loss_mse(x, x_hat)
    # hxy = conditional_entropy(z_std)
    mmd = loss_mmd(z_mu, x_hat, z_mu, use_cuda)
    elbo = loss_elbo(z_mu, z_std)
    total_loss = mse + (beta * elbo + alpha) * gamma

    nan_check_and_break(mse, 'mse')
    nan_check_and_break(elbo, 'elbo')
    nan_check_and_break(mmd, 'mmd')
    nan_check_and_break(total_loss, 'total_loss')
    return total_loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.n_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.n_bad_epochs = 0
            self.best = metrics
        else:
            self.n_bad_epochs += 1

        if self.n_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode' + mode + ' unknown')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def gauss_kernel(size=5, sigma=1.0,n_channels=1, cuda=False):
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return kernel

def conv_gauss(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    conv_img = F.conv2d(img, kernel, groups=n_channels)
    return conv_img

def laplacian_pyramid(img, kernel, max_levels=5):
    current_img = img
    pyramid_img = []
    
    for level in range(max_levels):
        conv_img = conv_gauss(current_img, kernel)
        diff = current_img - conv_img
        pyramid_img.append(diff)
        current_img = F.avg_pool2d(conv_img, 2)
    
    pyramid_img.append(current_img)
    return pyramid_img

def laploss(img1,img2, max_levels=3, kernel=None, k_size=5, sigma=2.0):
    
    if kernel is None or kernel.shape[1] != img1.shape[1]:
        kernel = gauss_kernel(size=k_size, sigma=sigma,n_channels=img1.shape[1], cuda=img1.is_cuda)
    pyramid_img1 = laplacian_pyramid(img1, kernel,max_levels)
    pyramid_img2 = laplacian_pyramid(img2, kernel, max_levels)
    return sum(F.l1_loss(a, b) for a, b in zip(pyramid_img1, pyramid_img2))
