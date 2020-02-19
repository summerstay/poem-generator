# poem-generator
Generates rhyming poetry using Huggingface GPT-2

import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn.functional as F

import pickle  

import copy

To get started quickly, replace the line 

model = GPT2LMHeadModel.from_pretrained("poetry")

with

model = GPT2LMHeadModel.from_pretrained("gpt2")

If you use Gwern's finetuned poetry GPT-2 model, the results are better, though.

