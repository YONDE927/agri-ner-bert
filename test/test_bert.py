import torch
from transformers import BertForTokenClassification, AutoTokenizer, BertJapaneseTokenizer 
import sys
import os 

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),'src'))
from util import label2id

model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese",num_labels=3)
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
#サブワードの分割をするかどうか
#tokenizer.do_subword_tokenize=False

line = "温暖な気候を生かし輸出向けのサン富士リンゴ育種に力を注いでいる。輸出先の食味やニーズに合わせ、新品種を育成した。"

inputs = tokenizer(line,return_tensors="pt")
print(inputs)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#print(tokens)
for i,t in enumerate(tokens):
    print(i,':',t,end='  ')
begin_id = int(input("\nbegin id?:"))
end_id = int(input("end id?:"))

label=[]
flg=False
print(begin_id,end_id)
for i,t in enumerate(tokens):
    if(begin_id == i):
        label.append('B_MEI')
        if(end_id != i):
            flg=True
    elif(end_id == i):
        label.append('I_MEI')
        flg=False
    else:
        if(flg):
            label.append('I_MEI')
        else:
            label.append('O')

print('label:\n',label)
#print(tokenizer.decode(inputs['input_ids'][0]))

outputs = model(**inputs,labels=label2id(label))

print(outputs)
