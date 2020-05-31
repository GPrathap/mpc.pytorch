import torch
import torch.nn.functional as F
import torch.autograd as tag

class SquareAndMaxPool1d(tag.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, **kwargs):
        # we're gonna need indices for backward. Currently SquareAnd...
        # never actually returns indices, I left it out for simplicity
        kwargs['return_indices'] = True

        input_sqr = input ** 2
        output, indices = F.max_pool1d(input_sqr, kernel_size, **kwargs)
        ctx.save_for_backward(input, indices)

        return output, output

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        input, indices = ctx.saved_tensors

        # first we need to reconstruct the gradient of `max_pool1d`
        # by putting all the output gradient elements (corresponding to
        # input elements which made it through the max_pool1d) in their
        # respective places, the rest has gradient of 0. We do it by
        # scattering it against a tensor of 0s
        
        grad_output_unpooled = torch.zeros_like(input)
        grad_output_unpooled.scatter_(2, indices, grad_output1)

        # then incorporate the gradient of the "square" part of your
        # operator
        grad_input = 2. * input * grad_output_unpooled

        # the docs for backward
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function.backward
        # say that "it should return as many tensors, as there were inputs
        # to forward()". It fails to mention that if an argument was not a
        # tensor, it should return None (I remember reading this somewhere,
        # but can't find it anymore). Anyway, we need to
        # return a (grad_input, None) tuple to avoid a complaint that two
        # outputs were expected
        return grad_input,grad_input, None


f = SquareAndMaxPool1d.apply
xT = torch.randn(1, 1, 6, requires_grad=True, dtype=torch.float64)
tag.gradcheck(lambda t: f(t, 2), xT)