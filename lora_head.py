from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np
from peft import AutoPeftModelForCausalLM, PeftModelForTokenClassification
import peft
import os
from datasets import concatenate_datasets,DatasetDict
from itertools import islice
import json
import gzip
import torch.nn as nn
from typing import Any, List, Optional, Union
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = "bert-base-uncased"
lr = 1e-4
batch_size = 16
num_epochs = 500
load_model=False
#output_dir = 'kwwww/test_16_2000'
r_=1
num_heads = 12
use_headwise = False

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

imdb['train'] = imdb_split_1000
imdb['test'] = imdb_split_test

bionlp = imdb
print(bionlp)

split_size = bionlp['train'].shape[0]
output_dir =  f'kwwww/{model_checkpoint}_{r_}_{split_size}_headwise'
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
############################################
if use_headwise == True:
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        A_layers = nn.ModuleDict({})
        B_layers = nn.ModuleDict({})
        if r > 0:
            in_f = self.in_features
            out_f = int(self.out_features / num_heads)
            for i in range(num_heads):
                layerA = nn.Linear(in_f, r, bias=False)
                layerB = nn.Linear(r, out_f, bias=False)
                A_layers[f'{i}'] = layerA
                B_layers[f'{i}'] = layerB
                
            self.lora_A[adapter_name] = A_layers
            self.lora_B[adapter_name] = B_layers
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)
        
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            for i in self.lora_A[adapter_name].keys():
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(self.lora_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights=}")
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
        if adapter_name in self.lora_embedding_A.keys():
            for i in self.lora_A[adapter_name].keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name][i])
                nn.init.normal_(self.lora_embedding_B[adapter_name][i])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        
        A_weight_dict = self.lora_A[adapter]
        B_weight_dict = self.lora_B[adapter]
        delta_weight_trunck = []
        for i in A_weight_dict.keys():
            weight_A = A_weight_dict[i].weight
            weight_B = B_weight_dict[i].weight
            delta_weight_trunck.append(weight_B @ weight_A)
            
        print(delta_weight.shape)
        input()
        delta_weight = torch.cat(delta_weight_trunck, dim=2)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(delta_weight, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor
        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                delta_weight_trunck = []
                lora_A_dict = self.lora_A[active_adapter]
                lora_B_dict = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                for i in lora_A_dict.keys():
                    x = x.to(lora_A_dict[i].weight.dtype)
                    weight_A = lora_A_dict[i]
                    weight_B = lora_B_dict[i]
                    delta_weight_trunck.append(weight_B(weight_A(dropout(x))))
                delta_weight = torch.cat(delta_weight_trunck, dim=2)
                result += delta_weight * scaling

        result = result.to(previous_dtype)
        return result
    peft.tuners.lora.layer.LoraLayer.reset_lora_parameters = reset_lora_parameters
    peft.tuners.lora.layer.LoraLayer.update_layer = update_layer
    peft.tuners.lora.layer.Linear.get_delta_weight = get_delta_weight
    peft.tuners.lora.layer.Linear.forward = forward
############################################
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=r_, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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
        pass
            
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
