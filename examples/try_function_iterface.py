import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    no_op_forward = True
    def __init__(current_x=None, current_u=None, verbose=0, back_eps=1e-3, no_op_forward=False):
        print("===============", MyReLU.no_op_forward)
        MyReLU.no_op_forward = False
    
    options_class = ""
    @staticmethod
    def forward(self, input, options):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        MyReLU.options_class = options
        if MyReLU.no_op_forward :
            self.save_for_backward(input)
        MyReLU.options_class['p'] = 4567
        return input.clamp(min=0), input.clamp(min=0), input.clamp(min=0)

    @staticmethod
    def backward(self, grad_output, grad_output1, grad_output2):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        print(MyReLU.options_class['p'])
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        # print("MyReLU.options_class['p']: ", MyReLU.options_class['p'])
        return grad_input, None


dtype = torch.float
device = torch.device("cuda")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

options = {}
options["m1"] = True

for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu_me = MyReLU

    relu = relu_me.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    r1, r2, r3 = relu(x.mm(w1), options)
    y_pred = r1.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
        print("MyReLU.options_class['p']: ", relu_me.options_class['p'])

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()