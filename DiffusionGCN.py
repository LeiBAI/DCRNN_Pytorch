import torch
import torch.nn as nn

#according to GraphWave Net implementation
class DiffusionGCN(nn.Module):
    def __init__(self, supports, node_num, dim_in, dim_out, order, kernel='conv'):
        #order must be integer
        super(DiffusionGCN, self).__init__()
        self.node_num = node_num
        self.supports = supports
        self.supports_len = len(supports)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.order = order
        self.kernel = kernel
        if kernel == 'mlp':
            self.mlp = nn.Linear(dim_in*(order*self.supports_len+1), dim_out)
        elif kernel == 'conv':
            self.mlp = torch.nn.Conv1d(dim_in*(order*self.supports_len+1), dim_out,
                                        kernel_size=1)
        else:
            raise ValueError('Kernel is not defined')
    def forward(self, x):
        #shape of x is [B, N, D]
        batch_size = x.shape[0]
        #print(x.shape[1] , self.node_num , self.dim_in , x.shape[2])
        assert x.shape[1] == self.node_num and self.dim_in == x.shape[2]

        out = [x]
        for support in self.supports:
            #x1 = torch.sparse.mm(supports, x0)
            x1 = torch.einsum('ij, bjk -> bik', support, x)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = torch.einsum('ij, bjk -> bik', support, x1)
                out.append(x2)
                x1 = x2
        out = torch.cat(out,dim=-1)     #B, N, D, order
        if self.kernel == 'mlp':
            out = out.reshape(batch_size*self.node_num, -1)     #B*N, D
            out = self.mlp(out)
            out = out.reshape(batch_size, self.node_num, self.dim_out)
        elif self.kernel == 'conv':
            out = out.reshape(batch_size, self.node_num, -1)
            out = out.permute(0, 2, 1)      #B, D, N
            out = self.mlp(out)
            out = out.permute(0, 2, 1)      #B, N, D
        return out

#according to DCRNN pytorch implementation
class DiffusionGCN2(nn.Module):
    def __init__(self, supports, node_num, dim_in, dim_out, order, kernel='conv'):
        #order must be integer
        super(DiffusionGCN2, self).__init__()
        self.node_num = node_num
        self.supports = supports
        self.supports_len = len(supports)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.order = order
        self.kernel = kernel
        self.weight = nn.Parameter(torch.FloatTensor(size=(dim_in*(order*self.supports_len+1), dim_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(dim_out,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=0.)

    def forward(self, x):
        #shape of x is [B, N, D]
        batch_size = x.shape[0]
        #print(x.shape[1] , self.node_num , self.dim_in , x.shape[2])
        assert x.shape[1] == self.node_num and self.dim_in == x.shape[2]

        out = [x]
        x0 = x
        for support in self.supports:
            x1 = torch.einsum('ij, bjk -> bik', support, x0)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = 2 * torch.einsum('ij, bjk -> bik', support, x1) - x0
                out.append(x2)
                x1, x0 = x2, x1
        out = torch.cat(out,dim=-1)     #B, N, D, order
        out = out.reshape(batch_size*self.node_num, -1)     #B*N, D
        out = torch.matmul(out, self.weight)  # (batch_size * self._num_nodes, output_size)
        out = torch.add(out, self.biases)
        out = out.reshape(batch_size, self.node_num, self.dim_out)
        return out

