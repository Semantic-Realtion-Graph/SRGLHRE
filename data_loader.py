import os
import csv
import copy
import json
import logging
import time
import torch
from torch.utils.data import TensorDataset
import random
from utils import load_entity2id,write_entity2id

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label,false_labels):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.false_labels = false_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id,
                 e1_mask, e2_mask,e1_id,e2_id,false_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.false_labels = false_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        #self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
            
        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)
class TacredProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        #self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    
    def _preprocess_example(self,text_a):
        for i,text in enumerate(text_a):
            flag = True
            if len(text) >= 5:
                length_text = len(text) - 1
                index = 0
                while (index < length_text - 1):
                    if text[index] != text[index + 1]:
                        flag=False
                        break
                    else:
                        index += 1
                if flag:
                    text_a[i] = text[0:3]
        return text_a
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = json.loads(line[0])
            #text_a = ' '.join(line['tokens'])
            text_a = line['tokens']
            text_a = self._preprocess_example(text_a)

            label = self.relation_labels.index(line['label'])
            entity_pos = line['entities']
            
            # 假设entity之间不重叠
            if entity_pos[0][0] > entity_pos[0][1]:
                tmp = entity_pos[0][0]
                entity_pos[0][0] = entity_pos[0][1]
                entity_pos[0][1] = tmp
            if entity_pos[1][0] > entity_pos[1][1]:
                tmp = entity_pos[1][0]
                entity_pos[1][0] = entity_pos[1][1]
                entity_pos[1][1] = tmp
            #entity_pos = sorted(entity_pos)
            entity1_start, entity1_end = entity_pos[0][0], entity_pos[0][1] 
            entity2_start, entity2_end = entity_pos[1][0], entity_pos[1][1]
            
            
            if entity1_start < entity2_start :     
                text_a.insert(entity1_start, '<e1>') 
                
                text_a.insert(entity1_end+1, '</e1>')
                
                text_a.insert(entity2_start+2, '<e2>')
                
                text_a.insert(entity2_end+3,'</e2>')
                
            
            else:
                text_a.insert(entity2_start, '<e2>') 
                
                text_a.insert(entity2_end+1, '</e2>')
                
                text_a.insert(entity1_start+2, '<e1>')
                
                text_a.insert(entity1_end+3,'</e1>')
                
            text = ' '.join([text_a[i] for i in range(len(text_a))])
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
            
        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)


class FewRelProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args,task_index):
        self.args = args
        self.train_file = os.path.join(self.args.data_dir, "train/train_"+str(task_index)+".jsonl")
        self.eval_file = os.path.join(self.args.data_dir, "val/val_" + str(task_index) + ".jsonl")
        self.test_file = os.path.join(self.args.data_dir, "val/test_all.jsonl")
        # self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

# 删减重复的多个字符，如 “-----------” -> "---"
    def _preprocess_example(self, text_a):
        for i, text in enumerate(text_a):
            flag = True
            if len(text) >= 5:
                length_text = len(text) - 1
                index = 0
                while (index < length_text - 1):
                    if text[index] != text[index + 1]:
                        flag = False
                        break
                    else:
                        index += 1
                if flag:
                    text_a[i] = text[0:3]
        return text_a

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = json.loads(line[0])
            # text_a = ' '.join(line['tokens'])
            text_a = line['tokens']
            text_a = self._preprocess_example(text_a)

            label = int(line['label'])
            if label>=80:
                print("label > = 80")
                print(label)
                print(line)
                time.sleep(1000)

            #print(line['false_labels'])
            if len(line['false_labels'])>10:
                print(line['false_labels'])
                print(line)
                time.sleep(1000)
            false_labels=[int(i) for i in line['false_labels']]
            for i in line['false_labels']:
                if int(i)>=80:
                    print("false label > = 80")
                    print(line['false_labels'])
                    print(line)
                    time.sleep(1000)
            entity_pos = [[0,0],[0,0]]
            entity_pos[0][0] = line["h"][2][0][0]
            entity_pos[0][1] = line["h"][2][0][-1]+1
            entity_pos[1][0] = line["t"][2][0][0]
            entity_pos[1][1] = line["t"][2][0][-1]+1
            entity1_start, entity1_end = entity_pos[0][0], entity_pos[0][1]
            entity2_start, entity2_end = entity_pos[1][0], entity_pos[1][1]

            if entity1_start < entity2_start:
                text_a.insert(entity1_start, '<e1>')
                text_a.insert(entity1_end + 1, '</e1>')
                text_a.insert(entity2_start + 2, '<e2>')
                text_a.insert(entity2_end + 3, '</e2>')
            else:
                text_a.insert(entity2_start, '<e2>')
                text_a.insert(entity2_end + 1, '</e2>')
                text_a.insert(entity1_start + 2, '<e1>')
                text_a.insert(entity1_end + 3, '</e1>')

            text = ' '.join([text_a[i] for i in range(len(text_a))])
            examples.append(InputExample(guid=guid, text_a=text, label=label,false_labels = false_labels))
        return examples
    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.train_file
        elif mode == 'dev':
            file_to_read = self.eval_file
        elif mode == 'test':
            file_to_read = self.test_file

        logger.info("LOOKING AT {}".format(file_to_read))
        return self._create_examples(self._read_tsv(file_to_read), mode)

processors = {
    "semeval": SemEvalProcessor,
    "tacred": TacredProcessor,
    "fewrel":FewRelProcessor
}


def convert_examples_to_features(examples, args, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True):
    '''
    :param examples: the examples of dataset
    :param args: parameters
    :param tokenizer:tokenizer
    :param cls_token: '[CLS]' marker
    :param cls_token_segment_id: The default is 0 if there is a sentence only
    :param sep_token:'[SEP]' marker
    :param pad_token: padding token,default is 0
    :return: features(InputFeatures)
    '''
    features = []
    max_seq_len = args.max_seq_len
    entity2id = load_entity2id(args.entity2id_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #tokens_a = tokenizer.tokenize(example)
        e11_p_ = example.text_a.index("<e1>")+5  # the start position of entity1
        e12_p_ = example.text_a.index("</e1>")  # the end position of entity1
        e21_p_ = example.text_a.index("<e2>")+5 # the start position of entity2
        e22_p_ = example.text_a.index("</e2>")  # the end position of entity2
        e1 = example.text_a[e11_p_:e12_p_].strip()
        e2 = example.text_a[e21_p_:e22_p_].strip()
        false_labels = example.false_labels
        #print(e1)
        if not entity2id.__contains__(e1):
            e1_id = len(entity2id)
            entity2id[e1] = e1_id
        else:
            e1_id = entity2id[e1]
        if not entity2id.__contains__(e2):
            e2_id = len(entity2id)
            entity2id[e2] = e2_id
        else:
            e2_id = entity2id[e2]


        tokens_a = tokenizer.tokenize(example.text_a)
        e11_p = tokens_a.index("<e1>")  # the start position of entity1
        e12_p = tokens_a.index("</e1>")  # the end position of entity1
        e21_p = tokens_a.index("<e2>")  # the start position of entity2
        e22_p = tokens_a.index("</e2>")  # the end position of entity2
        # Replace the token
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)
        
        if args.entity_start:
            e1_mask[e11_p] = 1
            e2_mask[e21_p] = 1
        else:
            ## entity average
            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1


        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)


        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask,
                          e1_id=e1_id,
                          e2_id=e2_id,
                          false_labels=false_labels))
    write_entity2id(entity2id,args.entity2id_file)
    return features


def load_and_cache_examples(args, test_list,task_index,tokenizer, mode):
    '''
    load examples from cache file
    :param args:  args
    :param test_list: test_list used to store seen examples
    :param task_index: task index of lifelong
    :param tokenizer:
    :param mode: Option is "train","test","dev","eval"
    :return:
    '''
    processor = processors[args.task](args,task_index)
    cached_features_file = os.path.join(
        args.data_dir,
         'cached_{}_{}_{}_{}_{}'.format(
            mode,
            args.task,
            task_index,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )
    if mode =="test":
        cached_features_file = os.path.join(
        args.data_dir,
         'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev" or mode=="eval":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args, tokenizer, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)

        torch.save(features, cached_features_file)

    # add the seen eval data to test_list
    if mode == "eval":
         test_list+=features
         features = test_list
    # if few_short, select samples from train features randomly
    elif mode == "train" :
        if args.few_short:
            features=random.sample(features,args.num_examples_per_task)

    # Convert to Tensors and build dataset

    return features


def convert_features_to_tensorDataset(features):
    '''
    convert features to TensorDataset
    :param features: features
    :return: dataset
    '''
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    all_e1_id = torch.tensor([f.e1_id for f in features], dtype=torch.long)
    all_e2_id = torch.tensor([f.e2_id for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_false_labels = torch.tensor([f.false_labels for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask, all_e1_id, all_e2_id,
                            all_false_labels)
    return dataset
