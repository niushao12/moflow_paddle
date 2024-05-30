import sys
import paddle
from mflow.models.basic import GraphLinear, GraphConv, ActNorm, ActNorm2D

class AffineCoupling(paddle.nn.Layer):

    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=
        False):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.layers = paddle.nn.LayerList()
        self.norms = paddle.nn.LayerList()
        self.mask_swap = mask_swap
        last_h = in_channel // 2
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
            vh = tuple(hidden_channels) + (in_channel // 2,)
        for h in vh:
            self.layers.append(paddle.nn.Conv2D(in_channels=last_h,
                out_channels=h, kernel_size=3, padding=1))
            self.norms.append(paddle.nn.BatchNorm2D(num_features=h))
            last_h = h

    def forward(self, input):
        in_a, in_b = input.chunk(chunks=2, axis=1)
        if self.mask_swap:
            in_a, in_b = in_b, in_a
        if self.affine:
            s, t = self._s_t_function(in_a)
            out_b = (in_b + t) * s
            logdet = paddle.sum(x=paddle.log(x=paddle.abs(x=s)).view(tuple(
                input.shape)[0], -1), axis=1)
        else:
            _, t = self._s_t_function(in_a)
            out_b = in_b + t
            logdet = None
        if self.mask_swap:
            result = paddle.concat(x=[out_b, in_a], axis=1)
        else:
            result = paddle.concat(x=[in_a, out_b], axis=1)
        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(chunks=2, axis=1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a
        if self.affine:
            s, t = self._s_t_function(out_a)
            in_b = out_b / s - t
        else:
            _, t = self._s_t_function(out_a)
            in_b = out_b - t
        if self.mask_swap:
            result = paddle.concat(x=[in_b, out_a], axis=1)
        else:
            result = paddle.concat(x=[out_a, in_b], axis=1)
        return result

    def _s_t_function(self, x):
        h = x
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = paddle.nn.functional.relu(x=h)
        h = self.layers[-1](h)
        s = None
        if self.affine:
            log_s, t = h.chunk(chunks=2, axis=1)
            s = paddle.nn.functional.sigmoid(x=log_s)
        else:
            t = h
        return s, t


class GraphAffineCoupling(paddle.nn.Layer):

    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True
        ):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']
        self.net = paddle.nn.LayerList()
        self.norm = paddle.nn.LayerList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(paddle.nn.BatchNorm1D(num_features=n_node))
            last_dim = out_dim
        self.net_lin = paddle.nn.LayerList()
        self.norm_lin = paddle.nn.LayerList()
        for out_dim in self.hidden_dim_linear:
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(paddle.nn.BatchNorm1D(num_features=n_node))
            last_dim = out_dim
        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim * 2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))
        out_17 = paddle.create_parameter(shape=paddle.zeros(shape=[1]).
            shape, dtype=paddle.zeros(shape=[1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=[1])))
        out_17.stop_gradient = not True
        self.scale = out_17
        mask = paddle.ones(shape=[n_node, in_dim])
        mask[masked_row, :] = 0
        self.register_buffer(name='mask', tensor=mask)

    def forward(self, adj, input):
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x)
        if self.affine:
            out = masked_x + (1 - self.mask) * (input + t) * s
            logdet = paddle.sum(x=paddle.log(x=paddle.abs(x=s)).view(tuple(
                input.shape)[0], -1), axis=1)
        else:
            out = masked_x + t * (1 - self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        s = None
        h = x
        for i in range(len(self.net)):
            h = self.net[i](adj, h)
            h = self.norm[i](h)
            h = paddle.nn.functional.relu(x=h)
        for i in range(len(self.net_lin) - 1):
            h = self.net_lin[i](h)
            h = self.norm_lin[i](h)
            h = paddle.nn.functional.relu(x=h)
        h = self.net_lin[-1](h)
        if self.affine:
            log_s, t = h.chunk(chunks=2, axis=-1)
            s = paddle.nn.functional.sigmoid(x=log_s)
        else:
            t = h
        return s, t


def test_AffineCoupling():
    from mflow.models.model import rescale_adj
    paddle.seed(seed=0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4
    adj = paddle.randint(low=0, high=2, shape=(bs, num_edge_type, nodes,
        nodes), dtype='float32')
    gc = AffineCoupling(in_channel=4, hidden_channels={512, 512}, affine=True)
    out = gc(adj)
    print('adj.shape:', tuple(adj.shape))
    print(tuple(out[0].shape), tuple(out[1].shape))
    r = gc.reverse(out[0])
    print(tuple(r.shape))
    print(r)
    print('paddle.abs(r-adj).mean():', paddle.abs(x=r - adj).mean())


def test_GraphAffineCoupling():
    from mflow.models.model import rescale_adj
    paddle.seed(seed=0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4
    x = paddle.randint(low=0, high=2, shape=(bs, nodes, ch), dtype='float32')
    adj = paddle.randint(low=0, high=2, shape=(bs, num_edge_type, nodes,
        nodes), dtype='float32')
    adj = rescale_adj(adj)
    in_dim = ch
    hidden_dim_dict = {'gnn': [8, 64], 'linear': [8]}
    gc = GraphAffineCoupling(nodes, in_dim, hidden_dim_dict, masked_row=
        range(0, nodes, 2), affine=True)
    out = gc(adj, x)
    print('in', tuple(x.shape), tuple(adj.shape))
    print(tuple(out[0].shape), tuple(out[1].shape))
    print(out)
    r = gc.reverse(adj, out[0])
    print(r)
    print(tuple(r.shape))
    print('paddle.abs(r-x).mean():', paddle.abs(x=r - x).mean())


if __name__ == '__main__':
    test_AffineCoupling()
