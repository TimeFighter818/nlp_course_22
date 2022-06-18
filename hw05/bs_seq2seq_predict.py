import torch
from torch.utils.data import DataLoader

from bs_seq2seq import BS_Seq2Seq_Data,Seq2Seq

from utils import WordTable

import os

import time
import logging

import yaml
import numpy as np



class BS_Config:
    def __init__(self, **data_dict):
        self.__dict__.update(data_dict)

with open("config.yaml","r",encoding="utf-8") as f:
    data_dict = yaml.load(f,Loader=yaml.FullLoader)
cfg = BS_Config(**data_dict)

word_table = WordTable()
word_table.load_dict()
print(len(word_table))

# 超参数
encoder_embedding_num = cfg.encoder_embedding_num
encoder_hidden_num = cfg.encoder_hidden_num
decoder_embedding_num = cfg.decoder_embedding_num
decoder_hidden_num = cfg.decoder_hidden_num
corpus_len = len(word_table)



device = "cuda" if torch.cuda.is_available() else "cpu"







if __name__ == "__main__":


    time_str = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'log/{time_str}.log',
        format='%(asctime)s - %(levelname)s: %(message)s'
    )



    corpus_src_path = "out/corpus_src.txt"
    with open(corpus_src_path,"r",encoding="utf-8") as fp:
        corpus_src = fp.readlines()

    corpus_tgt_path = "out/corpus_tgt.txt"
    with open(corpus_tgt_path,"r",encoding="utf-8") as fp:
        corpus_tgt = fp.readlines()

    corpus_max_len = 342

    bs_dataset = BS_Seq2Seq_Data(corpus_src, corpus_tgt, corpus_max_len)
    
    test_indexL = np.random.choice(range(len(bs_dataset)), 5)

    pth_path = "out/weights/Seq2Seq_time=20220618-201438_device=cuda_epoch=01_loss=0.9492.pth"

    model = Seq2Seq(encoder_embedding_num,encoder_hidden_num,decoder_embedding_num,decoder_hidden_num,corpus_len)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model = model.eval()
    model = model.to(device)

    
    for i in test_indexL:
    # for i in range(100):
    
        inputs, targets = bs_dataset[i]
        
    
        print("输入:",word_table.inputs2str(inputs.numpy()))
        print("理想输出:",word_table.inputs2str(targets.numpy()))
    


        
        inputs_tensor = inputs.view(1,-1)
        # print(inputs_tensor)
        predict = model.predict(inputs_tensor.to(device))

        # print(predict)
        print("预测输出:",word_table.inputs2str(predict))
