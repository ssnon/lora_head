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
batch_size = 16
num_epochs = 2000
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
output_dir =  f'kwwww/{model_checkpoint}-test_{r_}_{split_size}_head_revise'
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

############################################
class CustomBertSelfAttention(models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, original_layer):
        super(CustomBertSelfAttention, self).__init__(config)
        
        self.scaling = lora_alpha/r_
        
        self.query = original_layer.query
        self.key = original_layer.key
        self.value = original_layer.value
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.layerA_q = nn.Linear(config.hidden_size, r_).to(device)
        nn.init.normal_(self.layerA_q.weight, mean=0, std=0.01)
        nn.init.zeros_(self.layerA_q.bias)
        
        self.layerB_q = nn.Linear(r_, self.all_head_size).to(device)
        nn.init.zeros_(self.layerB_q.weight)
        nn.init.zeros_(self.layerB_q.bias)
            
        self.layerA_v = nn.Linear(config.hidden_size, r_).to(device)
        nn.init.normal_(self.layerA_v.weight, mean=0, std=0.01)
        nn.init.zeros_(self.layerA_v.bias)
        
        self.layerB_v = nn.Linear(r_, self.all_head_size).to(device)
        nn.init.zeros_(self.layerB_v.weight)
        nn.init.zeros_(self.layerB_v.bias)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        ############################################################
        new_query_layer_A = self.layerA_q(hidden_states)
        new_query = self.layerB_q(new_query_layer_A)
        new_value_layer_A = self.layerA_v(hidden_states)
        new_value = self.layerB_v(new_value_layer_A)
            
        new_combined_query = new_query * self.scaling
        new_combined_value = new_value * self.scaling
        
        ############################################################
        mixed_query_layer = self.query(hidden_states)
        ####
        mixed_query_layer = mixed_query_layer + new_combined_query
        ####

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            ###
            mixed_value_layer = self.value(hidden_states)
            mixed_value_layer = mixed_value_layer + new_combined_value
            value_layer = self.transpose_for_scores(mixed_value_layer)
            ###
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            ###
            mixed_value_layer = self.value(hidden_states)
            mixed_value_layer = mixed_value_layer + new_combined_value
            value_layer = self.transpose_for_scores(mixed_value_layer)
            ###

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
############################################

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

for param in model.parameters():
    param.requires_grad = False
    
for i in range(num_layer):
    model.bert.encoder.layer[i].attention.self = CustomBertSelfAttention(model.config, model.bert.encoder.layer[i].attention.self)

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
    model = model,
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
