import sys
import paddle
import numpy as np
from scipy import linalg as la
from utils import paddle_aux
logabs = lambda x: paddle.log(x=paddle.abs(x=x))


def conv_mask():
    pass


class ActNorm(paddle.nn.Layer):

    def __init__(self, in_channel, logdet=True):
        super().__init__()
        out_4 = paddle.create_parameter(shape=paddle.zeros(shape=[1,
            in_channel, 1, 1]).shape, dtype=paddle.zeros(shape=[1,
            in_channel, 1, 1]).numpy().dtype, default_initializer=paddle.nn
            .initializer.Assign(paddle.zeros(shape=[1, in_channel, 1, 1])))
        out_4.stop_gradient = not True
        self.loc = out_4
        out_5 = paddle.create_parameter(shape=paddle.ones(shape=[1,
            in_channel, 1, 1]).shape, dtype=paddle.ones(shape=[1,
            in_channel, 1, 1]).numpy().dtype, default_initializer=paddle.nn
            .initializer.Assign(paddle.ones(shape=[1, in_channel, 1, 1])))
        out_5.stop_gradient = not True
        self.scale = out_5
        self.register_buffer(name='initialized', tensor=paddle.to_tensor(
            data=0, dtype='uint8'))
        self.logdet = logdet

    def initialize(self, input):
        with paddle.no_grad():
            flatten = input.transpose(perm=[1, 0, 2, 3]).view(tuple(input.
                shape)[1], -1)
            mean = flatten.mean(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).unsqueeze(axis=3).transpose(perm=[1, 0, 2, 3])
            std = flatten.std(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).unsqueeze(axis=3).transpose(perm=[1, 0, 2, 3])
            paddle.assign(-mean, output=self.loc.data)
            paddle.assign(1 / (std + 1e-06), output=self.scale.data)

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(value=1)
        log_abs = logabs(self.scale)
        logdet = height * width * paddle.sum(x=log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNorm2D(paddle.nn.Layer):

    def __init__(self, in_dim, logdet=True):
        super().__init__()
        out_6 = paddle.create_parameter(shape=paddle.zeros(shape=[1, in_dim,
            1]).shape, dtype=paddle.zeros(shape=[1, in_dim, 1]).numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(paddle.
            zeros(shape=[1, in_dim, 1])))
        out_6.stop_gradient = not True
        self.loc = out_6
        out_7 = paddle.create_parameter(shape=paddle.ones(shape=[1, in_dim,
            1]).shape, dtype=paddle.ones(shape=[1, in_dim, 1]).numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(paddle.
            ones(shape=[1, in_dim, 1])))
        out_7.stop_gradient = not True
        self.scale = out_7
        self.register_buffer(name='initialized', tensor=paddle.to_tensor(
            data=0, dtype='uint8'))
        self.logdet = logdet

    def initialize(self, input):
        with paddle.no_grad():
            flatten = input.transpose(perm=[1, 0, 2]).view(tuple(input.
                shape)[1], -1)
            mean = flatten.mean(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).transpose(perm=[1, 0, 2])
            std = flatten.std(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).transpose(perm=[1, 0, 2])
            paddle.assign(-mean, output=self.loc.data)
            paddle.assign(1 / (std + 1e-06), output=self.scale.data)

    def forward(self, input):
        _, _, height = tuple(input.shape)
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(value=1)
        log_abs = logabs(self.scale)
        logdet = height * paddle.sum(x=log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(paddle.nn.Layer):

    def __init__(self, in_channel):
        super().__init__()
        weight = paddle.randn(shape=[in_channel, in_channel])
        q, _ = paddle.linalg.qr(x=weight)
        weight = q.unsqueeze(axis=2).unsqueeze(axis=3)
        out_8 = paddle.create_parameter(shape=weight.shape, dtype=weight.
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (weight))
        out_8.stop_gradient = not True
        self.weight = out_8

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        out = paddle.nn.functional.conv2d(x=input, weight=self.weight)
        res = paddle.linalg.slogdet(self.weight.squeeze().astype(dtype=
            'float64'))
        logdet = height * width * (res[0], res[1])[1].astype(dtype='float32')
        return out, logdet

    def reverse(self, output):
        return paddle.nn.functional.conv2d(x=output, weight=self.weight.
            squeeze().inverse().unsqueeze(axis=2).unsqueeze(axis=3))


class InvConv2dLU(paddle.nn.Layer):

    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = paddle.to_tensor(data=w_p)
        w_l = paddle.to_tensor(data=w_l)
        w_s = paddle.to_tensor(data=w_s)
        w_u = paddle.to_tensor(data=w_u)
        self.register_buffer(name='w_p', tensor=w_p)
        self.register_buffer(name='u_mask', tensor=paddle.to_tensor(data=
            u_mask))
        self.register_buffer(name='l_mask', tensor=paddle.to_tensor(data=
            l_mask))
        self.register_buffer(name='s_sign', tensor=paddle.sign(x=w_s))
        self.register_buffer(name='l_eye', tensor=paddle.eye(num_rows=tuple
            (l_mask.shape)[0]))
        out_9 = paddle.create_parameter(shape=w_l.shape, dtype=w_l.numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(w_l))
        out_9.stop_gradient = not True
        self.w_l = out_9
        out_10 = paddle.create_parameter(shape=logabs(w_s).shape, dtype=
            logabs(w_s).numpy().dtype, default_initializer=paddle.nn.
            initializer.Assign(logabs(w_s)))
        out_10.stop_gradient = not True
        self.w_s = out_10
        out_11 = paddle.create_parameter(shape=w_u.shape, dtype=w_u.numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(w_u))
        out_11.stop_gradient = not True
        self.w_u = out_11

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        weight = self.calc_weight()
        out = paddle.nn.functional.conv2d(x=input, weight=weight)
        logdet = height * width * paddle.sum(x=self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ (self.
            w_u * self.u_mask + paddle.diag(x=self.s_sign * paddle.exp(x=
            self.w_s)))
        return weight.unsqueeze(axis=2).unsqueeze(axis=3)

    def reverse(self, output):
        weight = self.calc_weight()
        return paddle.nn.functional.conv2d(x=output, weight=weight.squeeze(
            ).inverse().unsqueeze(axis=2).unsqueeze(axis=3))


class InvRotationLU(paddle.nn.Layer):

    def __init__(self, dim):
        super(InvRotationLU, self).__init__()
        weight = np.random.randn(dim, dim)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = paddle.to_tensor(data=w_p)
        w_l = paddle.to_tensor(data=w_l)
        w_s = paddle.to_tensor(data=w_s)
        w_u = paddle.to_tensor(data=w_u)
        self.register_buffer(name='w_p', tensor=w_p)
        self.register_buffer(name='u_mask', tensor=paddle.to_tensor(data=
            u_mask))
        self.register_buffer(name='l_mask', tensor=paddle.to_tensor(data=
            l_mask))
        self.register_buffer(name='s_sign', tensor=paddle.sign(x=w_s))
        self.register_buffer(name='l_eye', tensor=paddle.eye(num_rows=tuple
            (l_mask.shape)[0]))
        out_12 = paddle.create_parameter(shape=w_l.shape, dtype=w_l.numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(w_l))
        out_12.stop_gradient = not True
        self.w_l = out_12
        out_13 = paddle.create_parameter(shape=logabs(w_s).shape, dtype=
            logabs(w_s).numpy().dtype, default_initializer=paddle.nn.
            initializer.Assign(logabs(w_s)))
        out_13.stop_gradient = not True
        self.w_s = out_13
        out_14 = paddle.create_parameter(shape=w_u.shape, dtype=w_u.numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(w_u))
        out_14.stop_gradient = not True
        self.w_u = out_14

    def forward(self, input):
        bs, height, width = tuple(input.shape)
        weight = self.calc_weight()
        out = paddle.matmul(x=weight, y=input)
        logdet = width * paddle.sum(x=self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ (self.
            w_u * self.u_mask + paddle.diag(x=self.s_sign * paddle.exp(x=
            self.w_s)))
        return weight.unsqueeze(axis=0)

    def reverse(self, output):
        weight = self.calc_weight()
        return paddle.matmul(x=weight.inverse(), y=output)


class InvRotation(paddle.nn.Layer):

    def __init__(self, dim):
        super().__init__()
        weight = paddle.randn(shape=[dim, dim])
        q, _ = paddle.linalg.qr(x=weight)
        weight = q.unsqueeze(axis=0)
        out_15 = paddle.create_parameter(shape=weight.shape, dtype=weight.
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (weight))
        out_15.stop_gradient = not True
        self.weight = out_15

    def forward(self, input):
        _, height, width = tuple(input.shape)
        out = self.weight @ input
        res = paddle.linalg.slogdet(self.weight.squeeze().astype(dtype=
            'float64'))
        logdet = width * (res[0], res[1])[1].astype(dtype='float32')
        return out, logdet

    def reverse(self, output):
        return self.weight.squeeze().inverse().unsqueeze(axis=0) @ output


class ZeroConv2d(paddle.nn.Layer):

    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels=in_channel, out_channels=
            out_channel, kernel_size=3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        out_16 = paddle.create_parameter(shape=paddle.zeros(shape=[1,
            out_channel, 1, 1]).shape, dtype=paddle.zeros(shape=[1,
            out_channel, 1, 1]).numpy().dtype, default_initializer=paddle.
            nn.initializer.Assign(paddle.zeros(shape=[1, out_channel, 1, 1])))
        out_16.stop_gradient = not True
        self.scale = out_16

    def forward(self, input):
        out = paddle_aux._FUNCTIONAL_PAD(pad=[1, 1, 1, 1], value=1, x=input)
        out = self.conv(out)
        out = out * paddle.exp(x=self.scale * 3)
        return out


class GraphLinear(paddle.nn.Layer):
    """Graph Linear layer.
        This function assumes its input is 3-dimensional. Or 4-dim or whatever, only last dim are changed
        Differently from :class:`nn.Linear`, it applies an affine
        transformation to the third axis of input `x`.
        Warning: original Chainer.link.Link use i.i.d. Gaussian initialization as default,
        while default nn.Linear initialization using init.kaiming_uniform_

    .. seealso:: :class:`nn.Linear`
    """

    def __init__(self, in_size, out_size, bias=True):
        super(GraphLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = paddle.nn.Linear(in_features=in_size, out_features=
            out_size, bias_attr=bias)

    def forward(self, x):
        """Forward propagation.
            Args:
                x (:class:`chainer.Variable`, or :class:`numpy.ndarray`                or :class:`cupy.ndarray`):
                    Input array that should be a float array whose ``ndim`` is 3.

                    It represents a minibatch of atoms, each of which consists
                    of a sequence of molecules. Each molecule is represented
                    by integer IDs. The first axis is an index of atoms
                    (i.e. minibatch dimension) and the second one an index
                    of molecules.

            Returns:
                :class:`chainer.Variable`:
                    A 3-dimeisional array.

        """
        h = x
        h = h.reshape(-1, tuple(x.shape)[-1])
        h = self.linear(h)
        h = h.reshape(tuple(tuple(x.shape)[:-1] + (self.out_size,)))
        return h


class GraphConv(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        """

        :param in_channels:   e.g. 8
        :param out_channels:  e.g. 64
        :param num_edge_type:  e.g. 4 types of edges/bonds
        """
        super(GraphConv, self).__init__()
        self.graph_linear_self = GraphLinear(in_channels, out_channels)
        self.graph_linear_edge = GraphLinear(in_channels, out_channels *
            num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        """
        graph convolution over batch and multi-graphs
        :param h: shape: (256,9, 8)
        :param adj: shape: (256,4,9,9)
        :return:
        """
        mb, node, ch = tuple(h.shape)
        hs = self.graph_linear_self(h)
        m = self.graph_linear_edge(h)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type)
        m = m.transpose(perm=[0, 3, 1, 2])
        hr = paddle.matmul(x=adj, y=m)
        hr = hr.sum(axis=1)
        return hs + hr


def test_ZeroConv2d():
    in_channel = 1
    out_channel = 2
    x = paddle.ones(shape=[2, 1, 5, 5])
    net = ZeroConv2d(in_channel, out_channel)
    y = net(x)
    print('x.shape:', tuple(x.shape))
    print(x)
    print('y.shape', tuple(y.shape))
    print(y)


def test_actnorm():
    in_channel = 1
    out_channel = 2
    x = paddle.ones(shape=[2, 1, 3, 3])
    net = ActNorm(in_channel)
    y = net(x)
    print('x.shape:', tuple(x.shape))
    print(x)
    print('y.shape', tuple(y[0].shape))
    print(y[0])


if __name__ == '__main__':
    paddle.seed(seed=0)
    test_actnorm()
