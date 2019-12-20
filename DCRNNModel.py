import random
import torch
import torch.nn as nn
from Model.DCRNN1.DCRNNCell import DCGRUCell

class DCRNN(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order, num_layers=1):
        super(DCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DCGRUCell(supports, num_node, input_dim, hidden_dim, order))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(DCGRUCell(supports, num_node, hidden_dim, hidden_dim, order))

    def forward(self, x, init_state):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class Decoder(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order, num_layers):
        super(Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_node = num_node
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decoder_cells = nn.ModuleList()
        self.decoder_cells.append(DCGRUCell(supports, num_node, input_dim, hidden_dim, order))
        for _ in range(1, num_layers):
            self.decoder_cells.append(DCGRUCell(supports, num_node, hidden_dim, hidden_dim, order))
        self.projection = nn.Linear(hidden_dim, input_dim)
    def forward(self, inputs, init_state, teacher_forcing_ratio=0.5):
        # shape of inputs: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        #if teacher_forcing=1, then teacher forcing in all steps
        #if teacher_forcing=0, then no teacher forcing
        seq_length = inputs.shape[1]
        outputs = []
        current_input = inputs[:, 0, :, :self.input_dim]
        for t in range(seq_length-1):
            new_state = []
            for i in range(self.num_layers):
                state = init_state[i]
                state = self.decoder_cells[i](current_input, state)
                current_input = state
                new_state.append(state)
            init_state = torch.stack(new_state, dim=0)
            current_input = current_input.reshape(-1, self.hidden_dim) ## [B, N, dim_out] to [B*N, D]
            current_input = self.projection(current_input)
            current_input = current_input.reshape(-1, self.num_node, self.input_dim)
            outputs.append(current_input)
            #in the val and test phase, teacher_forcing_ratio=0
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[:, t+1, :, :self.input_dim] if teacher_force else current_input)
        return torch.stack(outputs, dim=1)      #B, T, N, dim_in

class DCRNNModel(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, out_dim, order, num_layers=1):
        super(DCRNNModel, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.output_dim = out_dim
        self.encoder = DCRNN(supports, num_node, input_dim, hidden_dim, order, num_layers)
        self.decoder = Decoder(supports, num_node, out_dim, hidden_dim, order, num_layers)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        _, encoder_hidden_state = self.encoder(source, init_state)
        GO_Symbol = torch.zeros(targets.shape[0], 1, self.num_node, self.input_dim).to(targets.device)
        targets_len = int(targets.shape[1])
        #targets = torch.cat([GO_Symbol, targets], dim=1)[:, 0:targets_len, ...]  # B, T, N, D to B, T, N, D
        targets = torch.cat([GO_Symbol, targets], dim=1)
        outputs = self.decoder(targets, encoder_hidden_state, teacher_forcing_ratio)
        return outputs      #B, T, N, D
        #return outputs[:, 1:, :, :]