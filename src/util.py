import torch

#label to id 
l2i_dic={
	"O":0,
	"B_MEI":1,
	"I_MEI":2
}

def label2id(x):
	global l2i_dic
	return torch.tensor([l2i_dic[y] for y in x])

def func1(word,names):
	for n in names:
		if(word in names):
			return True
	return False

def findspan(tokens,names):
	spans=[]
	flg=''
	begin=-1
	for name in names:
		for i,token in enumerate(tokens):
			tokenb=token.replace('#','')
			if tokenb in name:
				if begin<0:
					begin=i
				flg+=tokenb
			else:
				flg=''
				begin=-1
			if(flg==name):
				spans.append((begin,i,flg))
				begin=-1
	return spans

