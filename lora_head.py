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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = "bert-base-uncased"
lr = 1e-4
batch_size = 16
num_epochs = 500
load_model=False
#output_dir = 'kwwww/test_16_2000'
r_=1
num_heads = 12

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
class MyCustomLoraLayer(peft.tuners.lora.layer.LoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name, **kwargs) -> None:
        super().__init__(base_layer = base_layer, **kwargs)
        
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
        A_layers = []
        B_layers = []
        for i in range(num_heads):
            layerA = nn.Linear(self.in_features, r, bias=False)
            layerB = nn.Linear(r, self.out_features/num_heads, bias=False)
            A_layers.append(layerA)
            B_layers.append(layerB)
            
        self.A_q = nn.Sequential(*A_q)
        self.B_q = nn.Sequential(*B_q)
        
        if r > 0:
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
        
class MyCustomLinear(peft.tuners.lora.layer.Linear):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__(
            base_layer = base_layer,
            adapter_name =adapter_name,
            r = 1,
            lora_alpha = 1,
            lora_dropout = 0.0,
            fan_in_fan_out = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer = False,
            init_lora_weights = True,
            **kwargs,
        )
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
        
        A_weight_trunk = self.lora_A[adapter]
        B_weight_trunk = self.lora_B[adapter]
        delta_weight_trunck = []
        for i in range(num_heads):
            weight_A = A_weight_trunk[i].weight
            weight_B = B_weight_trunk[i].weight
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
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                for i in range(num_heads):
                    weight_A = lora_A[i].weight
                    weight_B = lora_B[i].weight
                    delta_weight_trunck.append(lora_B(lora_A(dropout(x))))
                result += delta_weight_trunck * scaling

        result = result.to(previous_dtype)
        return result
############################################
layerB = nn.Linear(123,124)
target = layerB
layerA = nn.Linear(123,124)
target2 = layerA

adapter_name = 'default'
#peft.tuners.lora.layer.LoraLayer = MyCustomLoraLayer(base_layer = target, adapter_name = adapter_name)
peft.tuners.lora.layer.Linear = MyCustomLinear(base_layer = target2 , adapter_name = adapter_name)

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
