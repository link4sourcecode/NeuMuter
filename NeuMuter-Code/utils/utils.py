import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
import pandas as pd
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_trainable_parameters(model):
  
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        if args.do_train:
            self.data = pd.read_csv(args.data_path)

        else:
            self.data = pd.read_csv(args.data_path_test)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        encoding = self.tokenizer.encode_plus(
            data['text'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }  


def get_attr_str(model_name):
    if 'gpt2' in model_name:
        attr_dict = {
            'transformer_layer': 'transformer.h',
            'ffn_out': 'mlp.c_proj',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    elif 'gpt-j' in model_name:
        attr_dict = {
            'transformer_layer': 'transformer.h',
            'ffn_out': 'mlp.fc_out',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    elif 'pythia' in model_name:
        attr_dict = {
            'transformer_layer': 'gpt_neox.layers',
            'ffn_out': 'mlp.dense_4h_to_h',
            'ffn_act': 'mlp.act',
            'lm_head': 'embed_out',
        }
    elif 'gpt-neo' in model_name:
        attr_dict = {
            'transformer_layer': 'transformer.h',
            'ffn_out': 'mlp.c_proj',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    elif 'phi' in model_name:
        attr_dict = {
            'transformer_layer': 'model.layers',
            'ffn_out': 'mlp.fc2',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    else:
        raise NotImplementedError(f"{model_name} attributes unkown!")
    return attr_dict

def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.h.0.mlp.c_proj')
        should return the same as model.transformer.h.0.mlp.c_proj
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def set_attributes(x: nn.Module, attributes: str, values):
    attr_list = attributes.split(".")
    for attr in attr_list[:-1]:
        x = getattr(x, attr)
    setattr(x, attr_list[-1], values)


def set_model_attributes(model, model_name):
    model.config.pad_token_id = model.config.eos_token_id
    model.attr_dict = get_attr_str(model_name)
    model.inner_dim = 4 * model.config.hidden_size
    if not hasattr(model.config, "n_layer"):
        model.config.n_layer = model.config.num_hidden_layers





def load_pretrained(args, load_model=True):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    if load_model:
        if 'gpt-j-6b' in args.model_name:
            # model = AutoModelForCausalLM.from_pretrained(args.model_name)
            cuda_list = '0, 1'.split(',')
            memory = '12.5GiB'
            cuda_memory = {int(cuda): memory for cuda in cuda_list}
            print(cuda_memory)
            # , output_hidden_states=True
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, config=config, device_map='auto')
        elif 'gpt-neo-125M' in args.model_name:
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            model = AutoModelForCausalLM.from_pretrained(args.model_name,pad_token_id=tokenizer.eos_token_id).to(device)

        elif 'gpt-neo-1.3B' in args.model_name:
            # config = transformers.AutoConfig.from_pretrained(args.model_name)
            # model = AutoModelForCausalLM.from_pretrained(args.model_name,pad_token_id=tokenizer.eos_token_id).to(device)
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, config=config, device_map='auto')
        elif 'gpt-neo-2.7B' in args.model_name:
            # config = transformers.AutoConfig.from_pretrained(args.model_name)
            # model = AutoModelForCausalLM.from_pretrained(args.model_name,pad_token_id=tokenizer.eos_token_id).to(device)
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, config=config, device_map='auto')
        elif 'tofu_ft_phi-1.5' in args.model_name:
            config = transformers.AutoConfig.from_pretrained(args.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, config=config, device_map='auto')

        set_model_attributes(model, args.model_name)


    else: model = None

    return tokenizer, model


