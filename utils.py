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
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
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
    assert len(preds) == len(labels)
    return acc_and_f1(task, preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(task, preds, labels, average='macro'):
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
        "micro_f1_no_relation": micro_f1_no_relation,
        "macro_f1_no_relation": macro_f1_no_relation,
    }


def score(key, prediction, no_relation, class_num, verbose=False):
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

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

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
    print("************* Writing entity_features ***************** ")
    with open(entity_feature_file, 'w') as file:
        json.dump(json_tensor2list(entity_features),file)
    file.close()  # 关闭文件




def load_edge_feature(edge_feature_file):
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
    print("************* Writing graph ***************** ")
    with open(graph_file, 'w') as file:
        json.dump(graph, file)
    file.close()


def load_entity2id(entity2id_file):
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


def update_graph(graph, e1_id, e2_id):
    if e1_id in graph.keys():
        if int(e2_id) not in graph[e1_id]:
            graph[e1_id].append(int(e2_id))
    else:
        graph[e1_id] = [int(e2_id)]
    if e2_id in graph.keys():
        if int(e1_id) not in graph[e2_id]:
            graph[e2_id].append(int(e1_id))
    else:
        graph[e2_id] = [int(e1_id)]
    return graph

def json_list2tensor(json_list):
    json_tensor={}
    for key in json_list.keys():
        json_tensor[key]=torch.tensor(json_list[key]).cuda().float()

    return json_tensor

def json_tensor2list(json_tensor):
    json_list={}
    for key in json_tensor.keys():
        json_list[key]=json_tensor[key].cpu().detach().numpy().tolist()
    return json_list

if __name__ == '__main__':
    write_features = {1: torch.Tensor([1,2,3,4,5,6]).numpy().astype(np.int).tolist(),
                      2: torch.Tensor([2,3,4,5,7,6]).numpy().astype(np.int).tolist()
                      }
    write_graph(write_features, "data/graph.json")
    graph = load_graph("data/graph.json")
    graph = update_graph(graph,"1","6")
    print(graph["1"])
    print(graph)
