from tkinter import N
import torch
import torch.nn as nn
from torch.utils.data import Dataset



class BS_Seq2Seq_Data(Dataset):
    def __init__(self, corpus_src, corpus_tgt, corpus_max_len):
        self.corpus_max_len = corpus_max_len  

        data_len = len(corpus_src)
        assert len(corpus_tgt) == data_len

        self.inputs = []
        self.targets = []

        for data_id in range(data_len):
            sentence_src = corpus_src[data_id].strip()
            sentence_tgt = corpus_tgt[data_id].strip()
            if sentence_src == "":
                continue 
            src_id = [int(i) for i in sentence_src.split(",")]
            tgt_id = [int(i) for i in sentence_tgt.split(",")]
            self.inputs.append(src_id)
            self.targets.append(tgt_id)
        
        self.data_len = len(self.inputs)
        assert len(self.targets) == self.data_len

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        # "1": "<PAD>",
        # "2": "<BOS>",
        # "3": "<EOS>",
        input = input + [1] * (self.corpus_max_len-len(input))
        target = [2]+ target + [3] + [1]*(self.corpus_max_len-2-len(target))


        input = torch.tensor(input)
        target = torch.tensor(target)
        return (input, target)
    
    def __len__(self):
        return self.data_len
    


class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len,encoder_embedding_num)
        self.fc = nn.Linear(encoder_embedding_num, encoder_hidden_num)
        self.trm_encoder = nn.TransformerEncoderLayer(d_model = encoder_hidden_num, nhead=8, batch_first=True)

    def forward(self,en_index):
        en_embedding = self.embedding(en_index)
        en_embedding = self.fc(en_embedding)
        encoder_hidden = self.trm_encoder(en_embedding)
        return encoder_hidden

class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num, decoder_hidden_num,corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len, decoder_embedding_num)
        self.fc = nn.Linear(decoder_embedding_num, decoder_hidden_num)
        self.trm_decoder = nn.TransformerDecoderLayer(d_model=decoder_hidden_num, nhead=8, batch_first=True)

    def forward(self,decoder_input,hidden):
        embedding = self.embedding(decoder_input)
        embedding = self.fc(embedding)
        decoder_output = self.trm_decoder(embedding, hidden)
        return decoder_output



class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,decoder_embedding_num,decoder_hidden_num,corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num,encoder_hidden_num,corpus_len)
        self.decoder = Decoder(decoder_embedding_num,decoder_hidden_num,corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num,corpus_len)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,inputs_tensor,targets_tensor):
        decoder_input = targets_tensor[:,:-1]
        label = targets_tensor[:,1:]

        encoder_hidden = self.encoder(inputs_tensor)
        decoder_output = self.decoder(decoder_input, encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))

        return loss
    
    def predict(self,inputs_tensor):
        result = []
        tensor_device = inputs_tensor.device
        encoder_hidden = self.encoder(inputs_tensor)
        decoder_hidden = encoder_hidden
        # "2": "<BOS>",

        decoder_input = torch.tensor([[2]],device=tensor_device)
        while True:
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            pre = self.classifier(decoder_output)

            w_index = int(torch.argmax(pre,dim=-1))

            # "3": "<EOS>",
            if w_index == 3 or len(result) > 133:
                break

            result.append(w_index)
            decoder_input = torch.tensor([[w_index]],device=tensor_device)
        return result


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_path = "out/corpus_onehot.txt"
    with open(data_path,"r",encoding="utf-8") as fp:
        textL = fp.readlines()
    
    bs_dataset = BS_Seq2Seq_Data(textL)
    bs_dataloader = DataLoader(bs_dataset,batch_size=128,shuffle=False)

    encoder_embedding_num = 50
    encoder_hidden_num = 100
    decoder_embedding_num = 107
    decoder_hidden_num = 100
    corpus_len = 45118
    device = "cuda" if torch.cuda.is_available() else "cpu"
