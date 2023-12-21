from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    BertConfig,
    BertModel,
    models
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np
from peft import AutoPeftModelForCausalLM, PeftModelForTokenClassification
import os
from datasets import concatenate_datasets,DatasetDict
from itertools import islice
import json
import gzip
import torch
import torch.nn as nn
import loralib as lora

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = "bert-base-uncased"
lr = 1e-4
batch_size = 8
num_epochs = 500
load_model=False
#output_dir = 'kwwww/test_16_2000'
r_=1
lora_alpha=16
num_layer=12
device = torch.device("cuda")

bionlp = load_dataset("imdb")

bionlp_test1 = bionlp['test'].shard(num_shards=5, index=0)#5000
bionlp_test2 = concatenate_datasets([bionlp_test1, bionlp['test'].shard(num_shards=5, index=1)])#10000
bionlp_test3 = concatenate_datasets([bionlp_test2, bionlp['test'].shard(num_shards=5, index=2)])#15000

bionlp_test_1 = bionlp['test'].shard(num_shards=5, index=3)
bionlp_test_2 = concatenate_datasets([bionlp_test_1, bionlp['test'].shard(num_shards=5, index=4)])#10000

bionlp['train'] = concatenate_datasets([bionlp['train'], bionlp_test3])#2000
bionlp['test'] = bionlp_test_2


imdb = bionlp

imdb_split_100 = imdb['train'].shard(num_shards=400, index=0) #100
imdb_split_200 = imdb['train'].shard(num_shards=200, index=0) #200
imdb_split_500 = imdb['train'].shard(num_shards=80, index=0) #500
imdb_split_1000 = imdb['train'].shard(num_shards=40, index=0) #1000
imdb_split_2000 = imdb['train'].shard(num_shards=20, index=0) #100
imdb_split_4000 = imdb['train'].shard(num_shards=10, index=0) #100
imdb_split_4500 = imdb['train'].shard(num_shards=9, index=0) #100
imdb_split_5000 = imdb['train'].shard(num_shards=8, index=0) #10
imdb_split_5800 = imdb['train'].shard(num_shards=7, index=0) #100
imdb_split_6500 = imdb['train'].shard(num_shards=6, index=0) #100
imdb_split_8000 = imdb['train'].shard(num_shards=5, index=0) #100
imdb_split_10000 = imdb['train'].shard(num_shards=4, index=0) #100

imdb_split_test = imdb['test'].shard(num_shards=4, index=0)

imdb['train'] = imdb_split_100
imdb['test'] = imdb_split_test

bionlp = imdb
print(bionlp)

split_size = bionlp['train'].shape[0]
output_dir =  f'kwwww/{model_checkpoint}_{r_}_{split_size}'
seqeval = evaluate.load("seqeval")
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"f1": f1, "accuracy": acc}
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

if load_model==True:
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
    
tokenized_bionlp = bionlp.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=r_, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print(model)
input()

if load_model==True:
    peft_model_id = output_dir
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
    model = PeftModel.from_pretrained(model, peft_model_id)
    print("----------------------------------------")
    
    for num in range(model.config.num_hidden_layers):
        lora_A_default_weights = model.base_model.model.bert.encoder.layer[num].attention.self.query.lora_A.default.weight
        lora_B_default_weights = model.base_model.model.bert.encoder.layer[num].attention.self.query.lora_B.default.weight
        print(num)
        print("LoRA A Default :", lora_A_default_weights)
        print("LoRA B Default :", lora_B_default_weights)
        input()
    print(model.state_dict())
    input()
    
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True
)

class MyCallback:
    def __init__(self, model):
        self.model = model
    def on_init_end(self, args, state, control, **kwargs):
        pass
    def on_train_begin(self, args, state, control, **kwargs):
        pass
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
    def on_step_begin(self, args, state, control, **kwargs):
        pass
    def on_step_end(self, args, state, control, **kwargs):
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        weight_dict = {}
        #for num in range(model.config.num_hidden_layers):
        num = 0
        lora_A_default_weights = model.base_model.model.bert.encoder.layer[num].attention.self.query.lora_A.default.weight
        lora_B_default_weights = model.base_model.model.bert.encoder.layer[num].attention.self.query.lora_B.default.weight
            
        #lora_A_default_weights = lora_A_default_weights.detach().cpu().numpy().tolist()
        #lora_B_default_weights = lora_B_default_weights.detach().cpu().numpy().tolist()
    
        index_a = f'weightA_{num}'
        index_b = f'weightB_{num}'
    
        weight_dict[index_a] = lora_A_default_weights
        weight_dict[index_b] = lora_B_default_weights
    
        weight_path = f'/mnt/hdd/weight/40000_20_2/weight_{epoch}.json'
        output_dir_1 = weight_path
        with open(output_dir_1, "w") as json_file:
            json.dump(weight_dict, json_file, indent=2, default=lambda x: x.tolist())
            
    def on_prediction_step(self, args, state, control, **kwargs):
        pass
    def on_log(self, args, state, control, **kwargs):
        pass
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    def on_save(self, args, state, control, **kwargs):
        pass
    def on_train_end(self, args, state, control, **kwargs):
        pass
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[MyCallback(model)],
)

trainer.train()

output_dir_1 = os.path.join(output_dir, "model_weights_epoch4.pth")
torch.save(model.state_dict(), output_dir_1)

trainer.push_to_hub()


REPO_NAME = output_dir # ex) 'my-bert-fine-tuned'
AUTH_TOKEN = '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' # <https://huggingface.co/settings/token>

model.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=AUTH_TOKEN
)
tokenizer.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=AUTH_TOKEN
)
