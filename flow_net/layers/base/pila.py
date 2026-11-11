import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



'''
By default, forward() and backward() of torch.autograd.Function
   is supplied with torch.no_grad() context.
   Hence, no need to worry about that inefficiency.
'''
class PilaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kabcdmn):
        k, a, b, c, d, m, n = kabcdmn
        negative = x.clamp_max(0)
        polynomial = ((a * negative + b) * negative + c) * negative + d
        y = torch.where(x >= 0, m * x + n, polynomial * torch.exp(k * negative))
        ctx.save_for_backward(x, kabcdmn)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, kabcdmn = ctx.saved_tensors
        k, a, b, c, d, m, n = kabcdmn
        kabcdmn_new = torch.stack([k, k*a, k*b+3*a, k*c+2*b, k*d+c, 0*k, m])
        return grad_output * PilaFunction.apply(x, kabcdmn_new), None


class Pila(nn.Module):
    def __init__(self, k=5, device=None):
        super(Pila, self).__init__()
        assert k > 0
        self.k = k

    def forward(self, x):
        k = self.k
        # Supplement B.1: native PyTorch avoids the missing pila_cpp extension.
        # Clamp the exponential's branch to avoid overflow for large positive x.
        negative = x.clamp_max(0)
        polynomial = (k**2 / 2 * negative - k) * negative**2 + negative
        return torch.where(x >= 0, x, polynomial * torch.exp(k * negative))

    def build_clone(self):
        return copy.deepcopy(self)




class CPila(nn.Module):
    def __init__(self, k=5, device=None):
        super(CPila, self).__init__()
        assert k > 0
        self.k = k
        self.pila = Pila(k=k, device=device)

    def forward(self, x):
        x = torch.cat((x-0.2, -x-0.2), 2)
        return self.pila(x) / 1.06

    def build_clone(self):
        return copy.deepcopy(self)

    def build_jvp_net(self, x):
        class CPilaJVP(nn.Module):
            def __init__(self, grad):
                super(CPilaJVP, self).__init__()
                self.register_buffer('grad', grad)

            def forward(self, x):
                return torch.cat((x,x), dim=2) * self.grad

        with torch.no_grad():
            y = self.forward(x)

            k = self.k
            a, b, c, d, m, n = k**2/2,-k,1,0,1,0
            kabcdmn = torch.tensor((k, k*a, k*b+3*a, k*c+2*b, k*d+c, 0*k, m), dtype=x.dtype, device=x.device)
            grad = torch.cat((pila_cpp.forward(x-0.2, kabcdmn), -pila_cpp.forward(-x-0.2, kabcdmn)), dim=2)
            grad.div_(1.06)

            return CPilaJVP(grad), y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.linspace(-5, +5, 500, dtype=torch.double).requires_grad_(True).cuda()
    #x = torch.randn(500, dtype=torch.double).requires_grad_(True).cuda()
    #x = x.view(5, 4, 25)

    y = Pila()(x)

    grad_x, = torch.autograd.grad(y.sum(), x, create_graph=True)
    import torchviz
    torchviz.make_dot((grad_x, x, y), params={"grad_x": grad_x, "x": x, "y": y}).render("torchviz", format="png")

    k = 5
    kabcdmn = torch.tensor((k,k**2/2,-k,1,0,1,0), dtype=torch.double, device=x.device)
    if torch.autograd.gradcheck(PilaFunction.apply, (x, kabcdmn)):
        print('Pila passed torch.autograd.gradcheck')
    if torch.autograd.gradgradcheck(PilaFunction.apply, (x, kabcdmn)):
        print('Pila passed torch.autograd.gradgradcheck')

    x = x.view(-1)
    y = y.view(-1)
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    plt.scatter(x_np,y_np)
    plt.title('Pila with k=5')
    plt.show()
