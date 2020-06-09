import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from torch.nn.modules.module import Module


PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RBERT, self).__init__(bert_config)
        self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.num_labels = bert_config.num_labels
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size*2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size*2, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 3, bert_config.num_labels, args.dropout_rate, use_activation=False)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    @staticmethod
    def update(e1_h,e2_h,cls_h,e1_id,e2_id,graph,edge_feature,entity_feature):

        '''
        update the SRG
        :param e1_h: hidden state of entity(vertex) 1
        :param e2_h: hidden state of entity(vertex) 2
        :param cls_h: hidden state of "[CLS]" (sentence)
        :param e1_id: the id of entity 1
        :param e2_id:the id of entity 2
        :param graph: SRG
        :param edge_feature: features of edges
        :param entity_feature:features of vertexs
        :return:
        '''


        # update graph,update entity_feature
        if graph.__contains__(str(e1_id)):
            if int(e2_id) not in graph[str(e1_id)]:
                graph[str(e1_id)].append(int(e2_id))
            #entity_feature[str(e1_id)] = ((np.array(entity_feature[str(e1_id)]) + e1_h.cpu().clone().detach().numpy())/2).tolist()
        else:
            graph[str(e1_id)] = [int(e2_id)]
        entity_feature[str(e1_id)] = e1_h.cpu().clone().detach().numpy().tolist()
        if graph.__contains__(str(e2_id)):
            if int(e1_id) not in graph[str(e2_id)]:
                graph[str(e2_id)].append(int(e1_id))
            #entity_feature[str(e2_id)] = ((np.array(entity_feature[str(e2_id)]) + e2_h.cpu().clone().detach().numpy())/2).tolist()
        else:
            graph[str(e2_id)] = [int(e1_id)]
        entity_feature[str(e2_id)] = e2_h.cpu().clone().detach().numpy().tolist()

        # update edge_feature
        #print("graph",graph)
        #print("entity_feature",entity_feature)
        e1_id_,e2_id_=e1_id,e2_id
        if e1_id>e2_id:
            e2_id_,e1_id_ = e1_id,e2_id
        edge=str(e1_id_)+"-"+str(e2_id_)
        #if edge in edge_feature:
            #edge_feature[edge]=((np.array(edge_feature[edge]) + cls_h.cpu().clone().detach().numpy())/2).tolist()
        #else:
        edge_feature[edge] = cls_h.cpu().clone().detach().numpy().tolist()
        #print(e1_id,e2_id,edge)
       


    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask,
                e1_ids,e2_ids,graph,edge_feature,entity_feature):
        """
        forward process of SRGLHRE
        process: ALBERT -> SRG -> classification
        :param input_ids,attention_mask,token_type_ids,labels,e1_mask,e2_mask,e1_ids,e2_ids,graph,edge_feature,entity_feature
        :return: logits, (hidden_states)
        """

        # get output from ALBERT(hidden states of tokens and CLS)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        
        length = len(e1_ids)
        batch_e1_h=torch.Tensor().cuda()
        batch_e2_h = torch.Tensor().cuda()

        # aggregate neighbor information from SRG
        for i in range(length):
            e1_id = int(e1_ids[i])
            e2_id = int(e2_ids[i])

            str_e1_id=str(e1_id)
            str_e2_id=str(e2_id)
            
            self.update(e1_h[i],e2_h[i],pooled_output[i],e1_id,e2_id,graph,edge_feature,entity_feature)
            h_e1_neighbor=[]
            h_e2_neighbor=[]
            e1_neighbors=graph[str_e1_id]
            e2_neighbors = graph[str_e2_id]
            h_e1_edges=[]
            h_e2_edges = []
            
            for e_id in e1_neighbors:
                str_e1_id=str(e1_id)
                str_e_id = str(e_id)
                h_e1_neighbor.append(entity_feature[str_e_id])
                if e_id<e1_id:
                    str_e1_id,str_e_id=str_e_id,str_e1_id
                h_e1_edges.append(edge_feature[str_e1_id+"-"+str_e_id])
            
            for e_id in e2_neighbors:
                str_e2_id=str(e2_id)
                str_e_id = str(e_id)
                h_e2_neighbor.append(entity_feature[str_e_id])
                #print(str_e2_id,str_e_id)
                if e_id<e2_id:
                    str_e2_id,str_e_id=str_e_id,str_e2_id
                h_e2_edges.append(edge_feature[str_e2_id+"-"+str_e_id])
            h_e1_neighbor=torch.from_numpy(np.array(h_e1_neighbor)).cuda().float()
            h_e2_neighbor = torch.from_numpy(np.array(h_e2_neighbor)).cuda().float()
            h_e1_edges = torch.from_numpy(np.array(h_e1_edges)).cuda().float()
            h_e2_edges = torch.from_numpy(np.array(h_e2_edges)).cuda().float()
            
            e1_h_ = attention(pooled_output[i],h_e1_edges,h_e1_neighbor)
            e2_h_ = attention(pooled_output[i],h_e2_edges,h_e2_neighbor)
            batch_e1_h=torch.cat((batch_e1_h, e1_h_), 0)
            batch_e2_h = torch.cat((batch_e2_h, e2_h_), 0)
        
        batch_e1_h = torch.cat((e1_h,batch_e1_h.view(length,-1)),1)
        batch_e2_h = torch.cat((e2_h,batch_e2_h.view(length,-1)),1)
        

            # Dropout -> relu -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(batch_e1_h)
        e2_h = self.e2_fc_layer(batch_e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
def attention(query, key, value):
    '''
    an attention mechanism to get weigth between vertexs in SRG
    :param query:
    :param key:
    :param value:
    :return:
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    result = torch.matmul(p_attn,value)

    return result




