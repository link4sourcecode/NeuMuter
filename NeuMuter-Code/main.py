
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.modeling_hardconcrete import *
import copy
from utils.NeuMuter import *
from utils.utils import MyCustomDataset, load_pretrained, seed_everything


# from test_unlearn import *
# from test_classification import *
# import warnings

# warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--model_name", default="gpt-neo-125M", type=str)
parser.add_argument("--data_path", default='./data/lm_extraction_32_0.csv', type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--do_train", default=True, type=bool)  # vocab_size = 0 means auto (for 
parser.add_argument("--max_length", default=200, type=int)
parser.add_argument("--input_length", default=200, type=int)
parser.add_argument("--output_length", default=200, type=int)
parser.add_argument("--padding_length", default=200, type=int)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--threshold", type=float, default=1e-1)
parser.add_argument("--stop_loss", type=float, default=8)
parser.add_argument("--out_dir", type=str, default='./Mask')
parser.add_argument("--start_mask_layer", type=int, default=1)
parser.add_argument("--mask_p", type=float, default=1, help="HC")
parser.add_argument("--beta", type=float, default=1/3, help="HC temperature")

parser.add_argument("--r", type=float, default=0.15, help="select threshold")
parser.add_argument("--lambda_l1", type=float, default=500)
parser.add_argument("--eta", type=float, default=5)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--per_nei_num", default=1)
parser.add_argument("--mask_prob", default=1)






def main(args):
    
    # load pretrained model and unlearned dataset
    seed_everything()
    tokenizer, model = load_pretrained(args)
    pretrained_model = copy.deepcopy(model)  
    dataset = MyCustomDataset(args, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # NeuMuter
    inject_mask(model, args)
    # Memorization Neuron Localization
    NeuMuter_localization(args, model, pretrained_model,tokenizer,dataset, data_loader, device)
    # Memorization Removal
    NeuMuter_removal(args, model)







    



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)