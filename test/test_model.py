import sys
import os 
import torch

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')),'src'))
from model import decoder

def main():
	preds=torch.tensor([0, 1, 2, 0])
	tokens=['pad','ブルー','##ベリー''pad']
	spans=decoder(preds,tokens)
	print(spans)

if __name__ == '__main__':
	main()