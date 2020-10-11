import os,sys
import numpy  as np 
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GRRN(nn.Module):
    def __init__(self,state_dim=10, edge_types = 5,time_step = 5):
        super(GRRN, self).__init__()
        self.time_step = time_step
        self.state_dim = state_dim
        self.edge_types = edge_types
        self.edge_fcs = nn.ModuleList()

        for i in range(self.edge_types):
            # incoming and outgoing edge embedding
            edge_fc = nn.Linear(self.state_dim, self.state_dim)
            self.edge_fcs.append(edge_fc)
            


        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid())
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid() )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh() )
        
        self.edge_attens = nn.ModuleList()

        for i in range(self.edge_types):
            edge_attention =  nn.Sequential(
                #nn.Dropout(),
                #nn.Linear(state_dim * 2, 4096),
                #nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(state_dim * 2, 1),
                nn.Sigmoid(),
                )
            self.edge_attens.append(edge_attention)

        self._initialization()


    # inputs with feature dim [batch, node_num, hidden_state_dim]
    # A with feature dim [batch, node_num, node_num]
    # reture output with feature dim [batch, node_num, output_dim]
    def forward(self,inputs,full_mask):
        node_num = inputs.size(1)

        full_mask = full_mask.view(-1,node_num,node_num,1).repeat(1,1,1,self.edge_types).float()
        full_mask = full_mask.detach()

        prop_state = inputs 

        all_scores = []

        for t in range(1 + self.time_step):
            
            message_states = []
            
            for i in range(self.edge_types):
                message_states.append(self.edge_fcs[i](prop_state))

            message_states_torch = torch.cat(message_states,dim=2).contiguous()
            message_states_torch = message_states_torch.view(-1,node_num * self.edge_types,self.state_dim)

            relation_scores = []

            for i in range(self.edge_types):
                relation_feature = message_states[i]
                feature_row_large = relation_feature.contiguous().view(-1,node_num,1,self.state_dim).repeat(1,1,node_num,1)
                feature_col_large = relation_feature.contiguous().view(-1,1,node_num,self.state_dim).repeat(1,node_num,1,1)
                feature_large = torch.cat((feature_row_large,feature_col_large),3)
                relation_score = self.edge_attens[i](feature_large)
                relation_scores.append(relation_score)

            graph_scores = torch.cat(relation_scores,dim=3).contiguous()
            graph_scores = graph_scores * full_mask
            all_scores.append(graph_scores)
            #graph_scores = graph_scores.detach()
            graph_scores = graph_scores.view(-1,node_num,node_num * self.edge_types)
            merged_message = torch.bmm(graph_scores, message_states_torch)


            a = torch.cat((merged_message,prop_state),2)

            r = self.reset_gate(a)
            z = self.update_gate(a)
            joined_input = torch.cat((merged_message, r * prop_state), 2)
            h_hat = self.tansform(joined_input)
            prop_state = (1 - z) * prop_state + z * h_hat

        return all_scores

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)
