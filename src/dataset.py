import torch
import torch.nn.functional as F
from tqdm import tqdm
from util import findspan

class AgriDataset(torch.utils.data.Dataset):
	def __init__(self,df,tokenizer):
		super(AgriDataset).__init__()
		print("-----building dataset...-----")
		self.tokenizer = tokenizer
		self.len = len(df)

		#preprocessing
		self.inputs=[]
		self.tokens=[]
		self.labels=[]
		for index, row in tqdm(df.iterrows()):
			#inputs
			inputs = self.tokenizer(row.sentence,return_tensors="pt")
			tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
			self.inputs.append(inputs)
			self.tokens.append(tokens)
			#print(tokens)
			#label
			spans=findspan(tokens,row.names.split(' '))
			label=[0]*len(tokens)
			for span in spans:
				b,e,_=span
				label[b]=1
				b+=1
				while(b<=e):
					label[b]=2
					b+=1
			#print(label)
			self.labels.append(torch.tensor(label))
		print("-----finish-----")

	def __len__(self):
		return self.len 

	def __getitem__(self,idx):
		return self.inputs[idx], self.labels[idx]

def collate_fn(batch):	
	max_size=max([t[1].size(0) for t in batch])
	first=True
	for inputs,labels in batch:
		if(max_size==labels.size(0)):
			ii= inputs['input_ids']
			tt= inputs['token_type_ids']
			am= inputs['attention_mask']
			lb=labels.unsqueeze(0)
		else:
			pad=torch.zeros(1,max_size-labels.size(0),dtype=torch.int)
			ii= torch.cat((inputs['input_ids'],pad),1)
			tt= torch.cat((inputs['token_type_ids'],pad),1)
			am= torch.cat((inputs['attention_mask'],pad),1)
			lb=torch.cat((labels,pad[0])).unsqueeze(0)
		if(first):
			x={
				'input_ids':ii,
				'token_type_ids':tt,
				'attention_mask':am
			}
			y=lb
			first=False
			continue
		x['input_ids']=torch.cat((x['input_ids'],ii))
		x['token_type_ids']=torch.cat((x['token_type_ids'],tt))
		x['attention_mask']=torch.cat((x['attention_mask'],am))
		y=torch.cat((y,lb))
	return x,y

