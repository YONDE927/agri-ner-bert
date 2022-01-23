import torch
import pandas as pd
from transformers import BertJapaneseTokenizer 
import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),'src'))
from dataset import AgriDataset,collate_fn
from util import findspan


df = pd.read_csv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),
				'data/train_sample.csv'),index_col=0)

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

dataset = AgriDataset(df,tokenizer)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=3,collate_fn=collate_fn)

inputs,labels = next(iter(dataloader))
print("inputs: \n",inputs)
print("labels: \n",labels)