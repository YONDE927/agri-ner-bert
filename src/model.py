import torch
import pandas as pd
from transformers import BertForTokenClassification, BertJapaneseTokenizer
import os,sys
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),'config'))
import config

def decoder(preds,tokens):
	spans=[]
	begin=0
	buf=None
	for i in range(len(tokens)):
		if(preds[i].item()==1):
			begin=i
			buf=tokens[i]
		elif(preds[i].item()==2):
			buf+=tokens[i].replace('#','')
		else:
			if buf:
				spans.append((begin,i-1,buf))
				begin=0
				buf=None
	if buf:
		spans.append((begin,i-1,buf))
	return spans

#トーカナイザ
def encoder(text):
	pass

#文章を入力して、品目を検出する
def nerscanner(text):
	pass