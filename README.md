# SRGLHRE
## Model Architecture
![](fig/modelv2-1.png)

## Dependencies

- perl (For evaluating official f1 score)
- python>=3.5
- torch>=1.1.0
- transformers>=2.3.0

## How to run
### Sem-Eval 2010 Task 8
```bash
python -u main.py \
    --task semeval \
    --model_type albert \
    --data_dir DATA_PATH \
    --do_train \
    --do_eval \
    --model_name_or_path MODEL_PATH \
    --model_dir ./model \
    --batch_size 16 \
    --max_seq_len 128 \
    --train_file train.tsv \
    --test_file test.tsv \
    --entity_start \
    --num_train_epochs 10
```
### TACRED
```bash
python -u main.py \
    --task tacred \
    --model_type albert \
    --data_dir DATA_PATH \
    --do_train \
    --do_eval \
    --model_name_or_path MODEL_PATH \
    --model_dir ./model \
    --batch_size 48 \
    --max_seq_len 128 \
    --train_file train.jsonl \
    --save_steps 10645 \
    --entity_start \
    --num_train_epochs 10
```
### FewRel
```bash
python -u main.py \
    --task fewrel \
    --model_type albert \
    --data_dir DATA_PATH \
    --do_train \
    --do_eval \
    --model_name_or_pathMODEL_PATH \
    --model_dir ./model \
    --batch_size 16 \
    --max_seq_len 128 \
    --entity_start \
    --per_memory_size 7\
    --few_short \
    --num_examples_per_task  100 \
    --num_train_epochs 10
```
## Parameters


| Parameters| Description|
|--|--|
| --task | the name of task|
|--model_type             |the name of pre-trained model|
|--data_dir               |the path of data|
|--do_train               |train the model|
|--do_eval                |evaluate the model|
|--model_name_or_path     |the path of the model to load|
|--batch_size             |batch size|
|--max_seq_len            |the max length of a sentence|
|--train_file             |the train data file|
|--test_file              |the evaluate data file|
|--entity_start           |use the hidden state of start markers <e1s> and <e2s> as the representation|
|--num_train_epochs       |the number of epoch|
|--few_short              |the few shot task|
|--num_examples_per_task  |limited supervision learning experiment|
  
