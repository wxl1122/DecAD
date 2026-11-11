import torch.nn as nn
from .layers.container import SequentialFlow
from .layers import *
from .layers import base
from enum import Enum
import numpy as np
from torch import Tensor
import torch
from typing import List

class Disentanglement(Enum):
    FBM = 1
    LBM = 2
    LCC = 3


ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base.Swish(),
    'LeakyLSwish': lambda b: base.LeakyLSwish(),
    'CLipSwish': lambda b: base.CLipSwish(),
    'ALCLipSiLU': lambda b: base.ALCLipSiLU(),
    'pila': lambda b: base.Pila(),
    'CPila': lambda b: base.CPila(),
    'lcube': lambda b: base.LipschitzCube(),
    'identity': lambda b: base.Identity(),
    'relu': lambda b: base.MyReLU(inplace=b),
    'CReLU': lambda b: base.CReLU(),
}


class FlowAssembly(SequentialFlow):

    def __init__(
            self,
            id,
            nflow_module,
            channel,
            coeff=0.9,
            n_lipschitz_iters=None,
            sn_atol=None,
            sn_rtol=None,
            n_power_series=5,
            n_dist='geometric',
            n_samples=1,
            activation_fn='LeakyLSwish',
            n_exact_terms=0,
            neumann_grad=True,
            grad_in_forward=False,
            nhidden=2,
            idim=64,
            densenet=False,
            densenet_depth=3,
            densenet_growth=32,
            learnable_concat=False,
            lip_coeff=0.98,

    ):
        chain = []

        def _quadratic_layer(channel):
            return InvertibleLinear(channel)

        def _actnorm(channel):
            return ActNorm(channel)

        def _lipschitz_layer():
            return base.get_linear

        def _iMonotoneBlock(preact=False):
            return iMonotoneBlock(
                FCNet(
                    preact=preact,
                    channel=channel,
                    lipschitz_layer=_lipschitz_layer(),
                    coeff=coeff,
                    n_iterations=n_lipschitz_iters,
                    activation_fn=activation_fn,
                    sn_atol=sn_atol,
                    sn_rtol=sn_rtol,
                    nhidden=nhidden,
                    idim=idim,
                    densenet=densenet,
                    densenet_depth=densenet_depth,
                    densenet_growth=densenet_growth,
                    learnable_concat=learnable_concat,
                    lip_coeff=lip_coeff,
                ),
                n_power_series=n_power_series,
                n_dist=n_dist,
                n_samples=n_samples,
                n_exact_terms=n_exact_terms,
                neumann_grad=neumann_grad,
                grad_in_forward=grad_in_forward,
            )
        chain.append(_iMonotoneBlock())
        chain.append(_actnorm(channel))
        chain.append(_iMonotoneBlock(preact=True))
        chain.append(_actnorm(channel))


        super(FlowAssembly, self).__init__(chain)

class FCNet(nn.Module):

    def __init__(
        self, preact, channel, lipschitz_layer, coeff, n_iterations, activation_fn, sn_atol, sn_rtol,
        nhidden=2, idim=64, densenet=False, densenet_depth=3, densenet_growth=32, learnable_concat=False, lip_coeff=0.98
    ):
        super(FCNet, self).__init__()
        nnet = []
        last_dim = channel
        if not densenet:
            if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                idim_out = idim // 2
                last_dim_in = last_dim * 2
            else:
                idim_out = idim
                last_dim_in = last_dim
            if(preact):
                nnet.append(ACT_FNS[activation_fn](False))
            for i in range(nhidden):
                nnet.append(
                    lipschitz_layer(
                        last_dim_in, idim_out, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                        atol=sn_atol, rtol=sn_rtol
                    )
                )
                nnet.append(ACT_FNS[activation_fn](True))
                last_dim_in = idim_out * 2 if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU'] else idim_out
            nnet.append(
                lipschitz_layer(
                    last_dim_in, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )
        else:
            first_channels = 64

            nnet.append(
                lipschitz_layer(
                    channel, first_channels, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

            total_in_channels = first_channels

            for i in range(densenet_depth):
                part_net = []

                # Change growth size for CLipSwish:
                if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                    output_channels = densenet_growth // 2
                else:
                    output_channels = densenet_growth

                part_net.append(
                    lipschitz_layer(
                        total_in_channels, output_channels, coeff=coeff, n_iterations=n_iterations, domain=2,
                        codomain=2, atol=sn_atol, rtol=sn_rtol
                    )
                )

                part_net.append(ACT_FNS[activation_fn](True))

                nnet.append(
                    LipschitzDenseLayer(ExtendedSequential(*part_net),
                                               learnable_concat,
                                               lip_coeff
                                               )
                )

                total_in_channels += densenet_growth

            nnet.append(
                lipschitz_layer(
                    total_in_channels, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

        self.nnet = ExtendedSequential(*nnet)

    def forward(self, x):
        y = self.nnet(x)
        return y

    def build_clone(self):
        class FCNetClone(nn.Module):
            def __init__(self, nnet):
                super(FCNetClone, self).__init__()
                self.nnet = nnet

            def forward(self, x):
                y = self.nnet(x)
                return y

        return FCNetClone(self.nnet.build_clone())

    def build_jvp_net(self, x):
        class FCNetJVP(nn.Module):
            def __init__(self, nnet):
                super(FCNetJVP, self).__init__()
                self.nnet = nnet

            def forward(self, v):
                jv = self.nnet(v)
                return jv

        nnet, y = self.nnet.build_jvp_net(x)
        return FCNetJVP(nnet), y
    


class DenoiseFlow(nn.Module):

    def __init__(
        self,
        disentangle=Disentanglement(1),
        pc_channel=3,

        # aug_channel=8,
        # n_aug = 4,
        aug_channel=768,

        n_injector = 7,
        num_neighbors = 32,

        cut_channel=0,
        nflow_module=2,
        coeff=0.9,
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        activation_fn='LeakyLSwish',
        n_exact_terms=0,
        neumann_grad=True,
        grad_in_forward=False,

        nhidden=2,
        idim=768,
        densenet=False,
        densenet_depth=3,
        densenet_growth=32,
        learnable_concat=False,
        lip_coeff=0.98,

    ):
        super(DenoiseFlow, self).__init__()

        self.disentangle = disentangle
        self.pc_channel = pc_channel

        self.aug_channel = aug_channel
        # self.n_aug = n_aug
        self.n_injector = n_injector
        self.num_neighbors = num_neighbors
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        self.coeff = coeff
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.activation_fn = activation_fn
        self.n_exact_terms = n_exact_terms
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.nhidden = nhidden
        self.idim = idim
        self.densenet = densenet
        self.densenet_depth = densenet_depth
        self.densenet_growth = densenet_growth
        self.learnable_concat = learnable_concat
        self.lip_coeff = lip_coeff

        self.dist = GaussianDistribution()

        flow_assemblies = []
        for i in range(self.nflow_module):
            flow = FlowAssembly(
                id=i,
                nflow_module=self.nflow_module,
                channel= self.aug_channel,
                coeff=self.coeff,
                n_lipschitz_iters=self.n_lipschitz_iters,
                sn_atol=self.sn_atol,
                sn_rtol=self.sn_rtol,
                n_power_series=self.n_power_series,
                n_dist=self.n_dist,
                n_samples=self.n_samples,
                activation_fn=self.activation_fn,
                n_exact_terms=self.n_exact_terms,
                neumann_grad=self.neumann_grad,
                grad_in_forward=self.grad_in_forward,
                nhidden=self.nhidden,
                idim=self.idim,
                densenet=self.densenet,
                densenet_depth=self.densenet_depth,
                densenet_growth=self.densenet_growth,
                learnable_concat=self.learnable_concat,
                lip_coeff=self.lip_coeff,
            )
            flow_assemblies.append(flow)
        self.flow_assemblies = nn.ModuleList(flow_assemblies)
        # -----------------------------------------------
        # Disentangle method
        if self.disentangle == Disentanglement.FBM:  # Fix binary mask
            # self.channel_mask = nn.Parameter(torch.ones((1, 1, self.pc_channel + self.aug_channel)),
            #                                  requires_grad=False)
            self.channel_mask = nn.Parameter(torch.ones((1, 1, self.aug_channel)),
                                             requires_grad=False)
            self.channel_mask[:, :, -self.cut_channel:] = 0.0

    def f(self, x: Tensor):
        B, N, _ = x.shape
        for i in range(self.nflow_module):
            x = self.flow_assemblies[i].forward(x)
        return x

    def g(self, z: Tensor):
        for i in reversed(range(self.nflow_module)):
            z = self.flow_assemblies[i].inverse(z)
        return z

    def log_prob(self, x: Tensor):
        z= self.f(x)#[B, N,  C]
        logp = 0
        return z, logp

    def sample(self, z: Tensor):
        full_x = self.g(z)
        # clean_x = full_x[..., :self.pc_channel]  # [B, N, 3]
        return full_x

    def forward(self, x: Tensor):
        #[BNC]
        z ,ldj= self.log_prob(x)
        # loss_denoise = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        # if self.disentangle == Disentanglement.FBM:  # Fix channel mask
        #     z[:, :, -self.cut_channel:] = 0
        predict_z = z

        predict_x = self.sample(predict_z)

        return predict_x


    def nll_loss(self, pts_shape, sldj):
        # ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj
        nll = -torch.mean(ll)
        return nll

    def denoise(self, noisy_pc: Tensor):
        clean_pc, _, _ = self(noisy_pc)
        return clean_pc

    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.nflow_module):
            self.flow_assemblies[i].chain[1].is_inited = True
            self.flow_assemblies[i].chain[3].is_inited = True



class Distribution:
    def log_prob(self, x: Tensor):
        raise NotImplementedError()
    def sample(self, shape, device):
        raise NotImplementedError()

# -----------------------------------------------------------------------------------------
class GaussianDistribution(Distribution):

    def log_prob(self, x: Tensor, means=None, logs=None):
        if means is None:
            means = torch.zeros_like(x)
        if logs is None:
            logs = torch.zeros_like(x)
        sldj = -0.5 * ((x - means) ** 2 / (2 * logs).exp() + np.log(2 * np.pi) + 2 * logs)
        sldj = sldj.flatten(1).sum(-1)
        return sldj

    def sample(self, shape, device):
        return torch.randn(shape, device=device)
    


class MaskLoss(nn.Module):

    def forward(self, mask):
        """
        mask: [1, 1, C]
        """
        loss = torch.abs(mask * (1 - mask))  # [1, 1, C]
        return torch.sum(loss)

# -----------------------------------------------------------------------------------------
class ConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.lossor = torch.nn.MSELoss()

    def forward(self, z1, z2):
        """
        z1: [B, N, C]
        z2: [B, N, C]
        """
        return self.lossor(z1, z2)