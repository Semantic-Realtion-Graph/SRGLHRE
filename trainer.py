import os
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from model import RBERT
from utils import set_seed, write_prediction, compute_metrics, get_label, MODEL_CLASSES, \
    load_entity_feature, load_entity2id, load_edge_feature, load_graph, write_edge_feature, write_entity_feature, \
    write_graph, write_entity2id

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        '''
        initial trainer
        :param args,train_dataset,dev_dataset, test_dataset
        '''
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
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

    def train(self):
        '''
        train process

        '''
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

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
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
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
                          'graph': self.graph,
                          'edge_feature': self.edge_feature,
                          'entity_feature': self.entity_feature
                          # 'entity2id': self.entity2id
                          }

                # get output from model
                outputs = self.model(**inputs)
                # get loss
                loss, logits = outputs[:2]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                # backward
                loss.backward()
                tr_loss += loss.item()
                # get prediction and labels
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                preds = np.argmax(preds, axis=1)

                #compute acc
                acc = (preds == out_label_ids).mean()

                # print logs information
                post_fix = {
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

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate('test')

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        '''
        eval process
        :param mode: "dev" or "test"
        :return:
        '''
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
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

        # logger.info("***** Eval results *****")
        # for key in sorted(results.keys()):
        # logger.info("  {} = {:.4f}".format(key, results[key]))
        output_eval_file = os.path.join("eval", "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for i in range(0, 10):
                writer.write("\n")
        return results

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
        """load model from model path"""
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
