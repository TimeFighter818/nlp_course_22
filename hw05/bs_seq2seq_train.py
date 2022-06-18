import torch
from torch.utils.data import DataLoader

from bs_seq2seq import BS_Seq2Seq_Data,Seq2Seq
from utils import WordTable

from tqdm import tqdm


import os

import time
import logging

import yaml

import random



class BS_Config:
    def __init__(self, **data_dict):
        self.__dict__.update(data_dict)

with open("config.yaml","r",encoding="utf-8") as f:
    data_dict = yaml.load(f,Loader=yaml.FullLoader)
cfg = BS_Config(**data_dict)

word_table = WordTable()
word_table.load_dict()


# 超参数
encoder_embedding_num = cfg.encoder_embedding_num
encoder_hidden_num = cfg.encoder_hidden_num
decoder_embedding_num = cfg.decoder_embedding_num
decoder_hidden_num = cfg.decoder_hidden_num
corpus_len = len(word_table)

batch_size = cfg.batch_size
num_epochs = cfg.num_epochs
learning_rate = cfg.learning_rate

device = "cuda" if torch.cuda.is_available() else "cpu"






if __name__ == "__main__":
    time_str = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'log/{time_str}.log',
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    save_text = f'out/weights/Seq2Seq_time={time_str}'+'_device={}_epoch={:02d}_loss={:.4f}.pth'

    corpus_src_path = "out/corpus_src.txt"
    with open(corpus_src_path,"r",encoding="utf-8") as fp:
        corpus_src = fp.readlines()

    corpus_tgt_path = "out/corpus_tgt.txt"
    with open(corpus_tgt_path,"r",encoding="utf-8") as fp:
        corpus_tgt = fp.readlines()

    corpus_max_len = 342

    bs_dataset = BS_Seq2Seq_Data(corpus_src, corpus_tgt, corpus_max_len)
    bs_dataloader = DataLoader(bs_dataset,batch_size=batch_size,shuffle=False,num_workers=4)



    model = Seq2Seq(encoder_embedding_num,encoder_hidden_num,decoder_embedding_num,decoder_hidden_num,corpus_len)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        val_loss = 0
        pbar = tqdm(bs_dataloader)

        for iteration,[inputs, targets] in enumerate(pbar):
        # for inputs,targets in bs_dataloader: 
            # print(inputs.shape,targets.shape)
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            loss = model(inputs,targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            val_loss = total_loss / (iteration + 1)
            

            desc_str = f"{'Train':8s} [{epoch + 1}/{num_epochs}] loss:{val_loss:.6f}"
            pbar.desc = f"{desc_str:40s}"
        
            
        model.eval()
        for _ in range(3):
            random_index = random.randint(0,len(textL)-1)
            test_inputs, test_targets = bs_dataset[random_index]

            test_inputs_tensor = test_inputs.view(1,-1)
            predict = model.predict(test_inputs_tensor.to(device))

            logging.info(f"random_index: {random_index}")
            logging.info(f"输入: {word_table.inputs2str(test_inputs.cpu().numpy())}")
            logging.info(f"理想输出: {word_table.inputs2str(test_targets.cpu().numpy())}")
            logging.info(f"预测输出: {word_table.inputs2str(predict)}")
        model.train()
            




        logging.info(f"epoch={epoch+1} total_loss={val_loss:.6f}")
        
        if not (epoch + 1) % 5 or (not epoch):
            torch.save(model.state_dict(), save_text.format(device, epoch + 1,val_loss) )
