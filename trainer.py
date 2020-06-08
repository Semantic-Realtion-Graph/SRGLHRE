import collections
import os
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from data_loader import convert_features_to_tensorDataset
from model import RBERT
from utils import set_seed, write_prediction, compute_metrics, get_label, MODEL_CLASSES, \
    load_entity_feature, load_entity2id, load_edge_feature, load_graph, write_edge_feature, write_entity_feature, \
    write_graph, write_entity2id, convert_inputs2InputFeatures, save_memory,split_data

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels,
                                                             finetuning_task=args.task)
        self.model = self.model_class(self.bert_config, args)
        self.graph = load_graph(args.graph_file)
        self.edge_feature = load_edge_feature(args.edge_feature_file)
        self.entity_feature = load_entity_feature(args.entity_feature_file)
        # self.entity2id = load_entity2id(args.entity2id_file)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self,train_dataset,task_index,memory):
        # train_sampler = RandomSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        random.shuffle(train_dataset)
        memory_list = [collections.OrderedDict() for i in range(8)]
        train_batch_size=self.args.batch_size
        if(task_index>0):
            train_batch_size=self.args.batch_size//2 
        num_train_epochs = self.args.num_train_epochs
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // (
                        len(train_dataset) // self.args.gradient_accumulation_steps**train_batch_size) + 1
        else:
            t_total = len(train_dataset) // self.args.gradient_accumulation_steps * num_train_epochs*train_batch_size

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        set_seed(self.args)

        train_dataset=split_data(train_dataset,train_batch_size)
        

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataset,desc="Iteration")
            for step, batch_train in enumerate(epoch_iterator):
                data=copy.copy(batch_train)
                if memory:
                    if(len(memory)<train_batch_size):
                        data.extend(memory)
                    else:
                        data.extend(random.sample(memory, train_batch_size))
                        random.shuffle(data)
                
                tensor_data = convert_features_to_tensorDataset(data)
                batch_iter = DataLoader(tensor_data, len(data))
                
                for batch in batch_iter:
                    preds = None
                    out_label_ids = None
                    self.model.train()
                    batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2],
                              'labels': batch[3],
                              'e1_mask': batch[4],
                              'e2_mask': batch[5],
                              'e1_ids': batch[6],
                              'e2_ids': batch[7],
                              "false_labels": batch[8],
                              'graph': self.graph,
                              'edge_feature': self.edge_feature,
                              'entity_feature': self.entity_feature
                              # 'entity2id': self.entity2id
                              }

                    outputs = self.model(**inputs)
                    loss, logits = outputs[:2]
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs['labels'].detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(
                            out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

                    preds_ = np.argmax(preds, axis=1)
                    

                    if(epoch == num_train_epochs-1):
                        for i in range(len(preds_)):
                            if preds_[i]>=task_index*8 and preds_[i]<(task_index+1)*8 and preds_[i]==out_label_ids[i]:
                                index = preds_[i]%8
                                if len(memory_list[index])<self.args.per_memory_size:
                                    memory_list[index][preds[i][preds_[i]]]=convert_inputs2InputFeatures(inputs,i)
                                else:
                                    min_key = min(memory_list[index].keys())
                                    if preds[i][preds_[i]]>min_key:
                                        del memory_list[index][min_key]
                                        memory_list[index][preds[i][preds_[i]]]=convert_inputs2InputFeatures(inputs,i)


                    acc = (preds_ == out_label_ids).mean()
                    post_fix = {
                        "task":task_index,
                        "epoch":epoch,
                        "iter": global_step,
                        "acc": acc,
                        "loss": loss.item()
                    }
                    logger.info(post_fix)
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1
                    if 0 < self.args.max_steps < global_step:
                        epoch_iterator.close()
                        break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        save_memory(memory, memory_list)
        return global_step, tr_loss / global_step

    def evaluate(self, mode,dataset,task_index):
       

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running %s on %s dataset *****"%(mode,mode))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5],
                          'e1_ids': batch[6],
                          'e2_ids': batch[7],
                          "false_labels":batch[8],
                          'graph': self.graph,
                          'edge_feature': self.edge_feature,
                          'entity_feature': self.entity_feature
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        write_prediction(self.args, os.path.join(self.args.eval_dir, "proposed_answers.txt"), preds)

        result = compute_metrics(self.args.task, preds, out_label_ids)
        results.update(result)


        output_eval_file = os.path.join("eval", "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** The %s result of task %d *****"%(mode,task_index))
            writer.write("***** The %s result of task %d ***** \n"%(mode,task_index))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for i in range(0, 10):
                writer.write("\n")
        return result
    
    
    

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        write_edge_feature(self.edge_feature, self.args.edge_feature_file)
        write_graph(self.graph, self.args.graph_file)
        write_entity_feature(self.entity_feature, self.args.entity_feature_file)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
