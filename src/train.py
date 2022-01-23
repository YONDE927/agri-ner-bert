import torch
from torch.optim import  Adam
import torch.nn as nn
import pandas as pd
from transformers import BertForTokenClassification, BertJapaneseTokenizer
import os,sys
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),'config'))
from dataset import AgriDataset,collate_fn
import config
from model import decoder

def main():
	df = pd.read_csv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),
				'data/train_sample.csv'),index_col=0)
	model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese",num_labels=config.NUM_LABELS)
	tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

	#データセットとローダーの初期化
	dataset = AgriDataset(df,tokenizer)
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=5,collate_fn=collate_fn)

	#最適化関数
	optimizer = Adam(model.parameters(),lr=config.LEARNING_RATE)

	#パラメーターの設定
	model.train()

	#学習
	for epoch in range(config.NUM_EPOCH):
		print("-----training epoch %d-----"%epoch)
		for inputs,labels in tqdm(dataloader):
			outputs = model(**inputs,labels=labels)
			loss = outputs.loss 
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		print("-----epoch %d finish-----"%epoch)

	#検証　未実装
	#モデルの出力テスト
	print("-----model output check-----")
	inputs,labels = dataset[0]
	with torch.no_grad():
		outputs = model(**inputs)
	logits = outputs.logits
	predictions = torch.argmax(logits, dim=-1)
	print("推論 正解 トークン ")
	for t,p,l in zip(dataset.tokens[0],predictions[0],labels):
		print(" {:<5d}{:<5d}{}".format(p.item(),l.item(),t))
	print("推論デコード")
	spans=decoder(predictions[0],dataset.tokens[0])
	print(spans)
	print("-----finish-----")
if __name__ == '__main__':
	main()
