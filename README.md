# SRGLHRE
## Model Architecture
![](fig/modelv2-1.png)

## Dependencies

- perl (For evaluating official f1 score)
- python>=3.5
- torch>=1.1.0
- transformers>=2.3.0

## How to run
### FewRel
```bash
python -u main.py \
    --task fewrel \
    --model_type albert \
    --data_dir DATA_PATH \
    --do_train \
    --do_eval \
    --model_name_or_path MODEL_PATH \
    --entity_start \
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
|--entity_start           |use the hidden state of start markers <e1s> and <e2s> as the representation|
|--few_shot               |if few shot, select some examples form dataset for limited supervision learning|
|--num_examples_per_task  |the number of examples selected in each task for limited supervision learning experiment|
    
    
## Data
We provide the lifelong FewRel benchamark which has been splited into ten tasks at ./data/fewrel

