import json
import os
import random
import logging
import time
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report, accuracy_score

from official_eval import official_f1
from model import RBERT
from collections import Counter

MODEL_CLASSES = {
    'bert': (BertConfig, RBERT, BertTokenizer),
    'roberta': (RobertaConfig, RBERT, RobertaTokenizer),
    'albert': (AlbertConfig, RBERT, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'
}

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    #get labels of dataset
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    # load tokenizer from ALBERT.tokenizer
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(task, preds, labels):
    '''
    compute acc and f1 from acc_and_f1()
    :param task: task of dataset
    :param preds: prediction
    :param labels: label
    :return:
    '''
    assert len(preds) == len(labels)
    return acc_and_f1(task, preds, labels)


def simple_accuracy(preds, labels):
    '''
    acc
    '''
    return (preds == labels).mean()


def acc_and_f1(task, preds, labels):
    '''
    the process of computing acc and f1
    :param task,preds,labels:
    :return: acc and f1
    '''
    acc = simple_accuracy(preds, labels)
    if (task == "semeval"):
        no_relation = 0
        class_num = 19
        # f1 = official_f1()
        pre, recall, f1 = score(labels, preds, no_relation, class_num)
        micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        labels_ = [i for i in range(class_num)]
        labels_.remove(no_relation)
        micro_f1_no_relation = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='micro')
        macro_f1_no_relation = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='macro')
    else:
        no_relation = 23
        class_num = 42
        pre, recall, f1 = score(labels, preds, no_relation, class_num)
        micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        labels_ = [i for i in range(class_num)]
        labels_.remove(no_relation)
        micro_f1_no_relation = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='micro')
        macro_f1_no_relation = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='macro')

    return {
        "acc": acc,
        "pre": pre,
        "recall": recall,
        "f1": f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_f1_no_relation": micro_f1_no_relation, # micro_f1 (reomving the other or no_relation label)
        "macro_f1_no_relation": macro_f1_no_relation, # macro_f1 (reomving the other or no_relation label)
    }


def score(key, prediction, no_relation, class_num):
    '''
    Detailed steps of computing f1
    :param key: true labels
    :param prediction: predict labels
    :param no_relation: the id of "no_relation" or "other"
    :param class_num: number of classes
    :return: pre ,recall and f1
    '''

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == no_relation and guess == no_relation:
            pass
        elif gold == no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
        elif gold != no_relation and guess == no_relation:
            gold_by_relation[gold] += 1
        elif gold != no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")

    prec_micro = 1.0
    recall_micro = 0.0

    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    prec_macro = 0.0
    recall_macro = 0.0

    # prec_macro = (np.array(list(correct_by_relation.values()))/np.array(list(guessed_by_relation.values()))).mean()
    # recall_macro=(np.array(list(correct_by_relation.values()))/np.array(list(gold_by_relation.values()))).mean()
    f1_macro = 0.0
    for item in gold_by_relation.keys():
        prec_macro_ = 1.0
        recall_macro_ = 0.0
        if item in correct_by_relation.keys():
            if item in guessed_by_relation.keys():
                prec_macro_ = correct_by_relation[item] / guessed_by_relation[item]
            recall_macro_ = correct_by_relation[item] / gold_by_relation[item]
        prec_macro = prec_macro + prec_macro_
        recall_macro = recall_macro + recall_macro_
        f1_macro = f1_macro + 2.0 * prec_macro_ * recall_macro_ / (prec_macro_ + recall_macro_)
    prec_macro = prec_macro / (class_num - 1)
    recall_macro = recall_macro / (class_num - 1)
    f1_macro = f1_macro / (class_num - 1)

    return prec_micro, recall_micro, f1_micro


def load_entity_feature(entity_feature_file):
    '''
    load entity features from entity_feature_file
    :param entity_feature_file: path of entity feature file
    :return: entity features
    '''
    print("************* Loading entity_features ***************** ")
    entity_features={}
    try:
        file = open(entity_feature_file, "r")
        entity_features = json.load(file)
        file.close()
    except json.decoder.JSONDecodeError:
        print("%s is empty!"%entity_feature_file)
    except FileNotFoundError:
        open(entity_feature_file, mode='w')
        print("%s 文件创建成功！"%entity_feature_file)
    return json_list2tensor(entity_features)



def write_entity_feature(entity_features, entity_feature_file):
    '''
        write entity features from entity_feature_file
        :param entity_feature_file: path of entity feature file
        :param: entity features: entity features
        '''
    print("************* Writing entity_features ***************** ")
    with open(entity_feature_file, 'w') as file:
        json.dump(json_tensor2list(entity_features),file)
    file.close()  # 关闭文件




def load_edge_feature(edge_feature_file):
    '''
        load edge features from edge_feature_file
        :param edge_feature_file: path of edge feature file
        :return: edge features
        '''
    print("************* Loading edge_features ***************** ")
    edge_feature={}
    try:
        file = open(edge_feature_file, "r")
        edge_feature = json.load(file)
        file.close()
    except json.decoder.JSONDecodeError:
        print("%s is empty!"%edge_feature_file)
    except FileNotFoundError:
        open(edge_feature_file, mode='w')
        print("%s 文件创建成功！"%edge_feature_file)
    return json_list2tensor(edge_feature)


def write_edge_feature(edge_features, edge_feature_file):
    print("************* Writing edge_features ***************** ")
    with open(edge_feature_file, 'w') as file:
        json.dump(json_tensor2list(edge_features), file)
    file.close()  # 关闭文件


def load_graph(graph_file):
    '''
          load graph  from graph_file
          :param graph_file: path of graph file
          :return: graph (adjacency list for vertexs)
          '''
    print("************* Loading graph ***************** ")
    graph={}
    try:
        file = open(graph_file, "r")
        graph = json.load(file)
        file.close()
    except json.decoder.JSONDecodeError:
        print("%s is empty!"%graph_file)
    except FileNotFoundError:
        open(graph_file, mode='w')
        print("%s 文件创建成功！"%graph_file)
    return graph


def write_graph(graph, graph_file):
    '''
    write graph to graph_file
    :param graph,graph_file
    '''
    print("************* Writing graph ***************** ")
    with open(graph_file, 'w') as file:
        json.dump(graph, file)
    file.close()


def load_entity2id(entity2id_file):
    '''
    load the map of entity to id
    :param entity2id_file: entity to id file path
    :return:
    '''
    print("************* Loading entity2id ***************** ")
    entity2id={}
    try:
        file = open(entity2id_file, "r")
        entity2id = json.load(file)
        file.close()
    except json.decoder.JSONDecodeError:
        print("%s is empty!"%entity2id_file)
    except FileNotFoundError:
        open(entity2id_file, mode='w')
        print("%s 文件创建成功！"%entity2id_file)
    return entity2id


def write_entity2id(entity2id, dicts_file):
    print("************* Writing entity2id ***************** ")
    with open(dicts_file, 'w') as file:
        json.dump(entity2id, file)
    file.close()


def json_list2tensor(json_list):
    '''
    convert json list to tensor
    '''
    json_tensor={}
    for key in json_list.keys():
        json_tensor[key]=torch.tensor(json_list[key]).cuda().float()

    return json_tensor

def json_tensor2list(json_tensor):
    '''
       convert tensor to json list
       '''
    json_list={}
    for key in json_tensor.keys():
        json_list[key]=json_tensor[key].cpu().detach().numpy().tolist()
    return json_list
