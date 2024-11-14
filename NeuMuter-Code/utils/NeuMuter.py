

import torch
import os
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer
import random
from torch.utils.data import DataLoader
from utils.modeling_hardconcrete import *
from utils.utils import set_model_attributes, get_attributes,  print_trainable_parameters



@torch.no_grad()
def get_sparsity(model, threshold):
    total, n = 0, 0
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        module = get_attributes(model, attr_str)
        mask = module.produce_mask(is_train_runtime=False).squeeze()
        total += (mask < threshold).sum().item()
        n += len(mask)
    return total / n
def compute_total_regularizer(model, start_layer_idx):
    total, n = 0, 0
    row_sparsity = []
    for module in model.modules():
        if hasattr(module, 'regularizer'):
            if module.layer_idx >= start_layer_idx:
                # total += module.regularizer()
                total += 1 - module.regularizer()
                # print(1-module.regularizer())
                # row_sparsity.append(module.regularizer())
                n += 1
    # return total / n,torch.var(torch.stack(row_sparsity))
    return total / n


def mask_text(search_model, search_tokenizer, text, mask_prob=0.8):
    # Tokenize the text
    tokens = search_tokenizer.tokenize(text)
    token_ids = search_tokenizer.convert_tokens_to_ids(tokens)

    # Determine the number of words to mask
    num_to_mask = max(1, int(len(tokens) * mask_prob))
    
    # Randomly select indices to mask
    mask_indices = random.sample(range(len(tokens)), num_to_mask)

    # Create a copy of tokens to modify
    modified_tokens = tokens[:]

    for idx in mask_indices:
        # Mask a single token
        masked_tokens = modified_tokens[:]
        masked_tokens[idx] = '[MASK]'
        
        # Tokenize masked text
        masked_token_ids = search_tokenizer.convert_tokens_to_ids(masked_tokens)
        masked_token_ids = search_tokenizer.build_inputs_with_special_tokens(masked_token_ids)
        masked_token_ids = torch.tensor(masked_token_ids).unsqueeze(0).to('cuda:0')

        # Predict masked token
        with torch.no_grad():
            outputs = search_model(masked_token_ids)
            predictions = outputs.logits

        # Get predicted token
        predicted_token_id = torch.argmax(predictions[0, idx+1]).item()
        predicted_token = search_tokenizer.convert_ids_to_tokens(predicted_token_id)

        # Replace the masked token with the predicted token
        modified_tokens[idx] = predicted_token

    predicted_text = search_tokenizer.convert_tokens_to_string(modified_tokens)
    return predicted_text


def generate_neighbour_data(dataset,tokenizer,device,args):
    search_tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
    search_model = BertForMaskedLM.from_pretrained('./bert-base-uncased')
    search_model = search_model.to('cuda:0')
    per_nei_num = 1
    neighbour_set = []
    for data in dataset:
        # print(data['input_ids'])
        
        
        inputs_text = tokenizer.decode(data['input_ids'])
        # print('inputs_text:',inputs_text)
        # break
        for i in range(args.per_nei_num):
            neighbour = {}
            test_text = mask_text(search_model,search_tokenizer,inputs_text,args.mask_prob)
            neighbour['input_ids']  = tokenizer(test_text, return_tensors='pt', padding='max_length', max_length=args.max_length, truncation=True).input_ids.to(device)
            # print(neighbour['input_ids'].shape)
            neighbour['attention_mask'] = torch.ones_like(neighbour['input_ids']).to(device)
            neighbour['labels'] = neighbour['input_ids'].to(device)
            neighbour_set.append(neighbour)
    print('neighbour_set:',len(neighbour_set))
    neighbour_data_loader = DataLoader(neighbour_set, batch_size=args.batch_size*per_nei_num, shuffle=True)
    return neighbour_data_loader

def compute_kl(pretrained_model, current_model, batch, device,T):
    """
    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.
        T: temperature.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits/T, -1)



    prob_q = torch.nn.functional.softmax(normal_outputs.logits/T, -1)


    log_prob_p = torch.log(prob_p + 1e-12)
    log_prob_q = torch.log(prob_q + 1e-12)
    loss = (prob_p * (log_prob_p - log_prob_q)).sum(-1).mean()


    return loss



@torch.no_grad()
def apply_neuron_mask(args, model, values,r):

    # First, set all to ones
    reinit_hardconcrete(model)
    set_mask_mode(model, is_train=False)
    total = 0
    n_neurons = []
    for l in range(args.start_mask_layer, model.config.n_layer):
        indices = torch.where(values[l] < r)[0] 
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        coef = get_attributes(model, attr_str)

        # coef.mask_scores[:,:,:,indices] = -100
        coef.mask_scores[:,:,:,indices] = torch.logit(values[l][indices])

        n = len(indices)
        total += n
        n_neurons.append(n)
        print(f"Layer {l} selected {n} neurons")

        
    print(f"Total neurons selected: {total}")

def inject_mask(model, args):
    set_model_attributes(model, args.model_name)
    patch_hardconcrete(model, args.model_name, mask_p=args.mask_p, beta=args.beta)


def NeuMuter_localization(args, model, pretrained_model,tokenizer,dataset, data_loader, device):
    

    model.eval()

    start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0

    # set tunable parameters
    print("Trainable Params:")
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if 'mask_score' in n:
            cnt += 1
            # if cnt > start_layer_idx and cnt < model.config.n_layer:
            if cnt > start_layer_idx :  
                p.requires_grad = True
                print(n, p.shape)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-"*100)
    print_trainable_parameters(model)

    neighbour_data_loader = generate_neighbour_data(dataset,tokenizer,device,args)





    # training
    set_mask_mode(model, is_train=True)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    model.zero_grad()
    reg_losses, lm_losses = [], []
    for i in range(args.epoch):
        epoch_lm_loss = 0.0 
        epoch_l1_loss = 0.0
        epoch_test_loss = 0.0 

        # for inputs in data_loader:
        for inputs , inputs_test in zip(data_loader,neighbour_data_loader):
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['labels'] = inputs['labels'].to(device)

            inputs_test['input_ids'] = inputs_test['input_ids'].to(device)
            inputs_test['attention_mask'] = inputs_test['attention_mask'].to(device)
            inputs_test['labels'] = inputs_test['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            lm_loss = outputs.loss
            
            lm_test_loss = compute_kl(pretrained_model, model, inputs_test, device,T=1)

            reg_loss= compute_total_regularizer(model, start_layer_idx)



            epoch_lm_loss += lm_loss.item()
            epoch_l1_loss += reg_loss.item()
            epoch_test_loss += lm_test_loss.item()


            if outputs.loss <= args.stop_loss:
                lambda_1 = -1
            else:
                lambda_1 = 5


            loss = lambda_1*lm_loss + args.lambda_l1 * reg_loss  + args.eta*lm_test_loss

            loss.backward()
            optimizer.step()
        l1_loss = epoch_l1_loss / len(data_loader)
        lm_loss = epoch_lm_loss / len(data_loader)
        test_loss = epoch_test_loss / len(data_loader)
        
        if (i+1) % 10 == 0:
            sparsity = get_sparsity(model, args.threshold)
            print(i+1, f'lm loss: {lm_loss:.3f}, reg_loss: {reg_loss:.3f}, test_loss: {test_loss:.6f}')
            print('  Sparsity:', sparsity)
            lm_losses.append(lm_loss)
            reg_losses.append(reg_loss)


    params = torch.sigmoid(torch.stack(params).squeeze()).detach().cpu()
    torch.save(params, os.path.join(args.out_dir, args.model_name +'_HC.pt'))



def NeuMuter_removal(args, model):
    attributions =  torch.load(os.path.join(args.out_dir, args.model_name +'_HC.pt'))
    args.inner_dim = model.inner_dim
    apply_neuron_mask(args, model,attributions,args.r)


