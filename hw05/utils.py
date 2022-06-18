from asyncio import current_task
import jieba
import json
import numpy as np

class WordTable:
    def __init__(self):
        self.word2id_dict = {}
        self.id2word_dict = {}
        self.max_sentence_len = 0

    def build(self, corpus_list):
        '''
            从语料库中建立词表
            保存词表并保存用One-hot表示的语料库
        '''
        word2id_dict = {
            "，":0,
            "<PAD>":1,
            "<BOS>":2,
            "<EOS>":3,
        }

        id2word_dict = {
            0:"，",
            1:"<PAD>",
            2:"<BOS>",
            3:"<EOS>",
        
        }
        
        corpus_src = []
        corpus_tgt = []
        assert len(id2word_dict) == len(word2id_dict)

        max_sentence_len = 0
        word_id = len(word2id_dict)
        jieba.load_userdict("user_dict.txt")
        for corpus in corpus_list:
            last_sentence = []
            for sentence in corpus:
                sentence_segL = jieba.cut(sentence)
                current_sentence = []
                for word in sentence_segL:
                    word = word.strip()
                    if word not in word2id_dict:
                        word2id_dict[word] = word_id
                        id2word_dict[word_id] = word
                        word_id += 1
                    current_sentence.append(str(word2id_dict[word]))
                if len(last_sentence)!=0 and len(current_sentence)!=0:
                    corpus_src.append(",".join(last_sentence)+"\n")
                    corpus_tgt.append(",".join(current_sentence)+"\n")
                last_sentence = current_sentence
                max_sentence_len = max(max_sentence_len, len(current_sentence))


        self.word2id_dict, self.id2word_dict, self.max_sentence_len = word2id_dict, id2word_dict, max_sentence_len

        with open("out/corpus_src.txt","w",encoding="utf-8") as fp:
            fp.writelines(corpus_src)
        with open("out/corpus_tgt.txt","w",encoding="utf-8") as fp:
            fp.writelines(corpus_tgt)
        with open("out/corpus_config.txt", "w", encoding="utf-8") as fp:
            fp.writelines("max_sentence_len: "+str(self.max_sentence_len))

        with open('out/word2id_dict.json','w',encoding="utf-8") as fp:
            json.dump(self.word2id_dict, fp,indent=4,ensure_ascii=False)
        with open('out/id2word_dict.json','w',encoding="utf-8") as fp:
            json.dump(self.id2word_dict, fp,indent=4,ensure_ascii=False)

    def load_dict(self):
        with open('out/word2id_dict.json','r',encoding="utf-8") as fp:
            self.word2id_dict = json.load(fp)
        with open('out/id2word_dict.json','r',encoding="utf-8") as fp:
            self.id2word_dict = json.load(fp)
    
    def word2id(self,word):
        return self.word2id_dict[word]
    
    def id2word(self,word_id):
        if type(word_id) is not str:
            word_id = str(word_id)
        return self.id2word_dict[word_id]

    # def prepare_gensim(self):
    #     with open("out/corpus_onehot.txt","r",encoding="utf-8") as fp:
    #         textL = fp.readlines()
        
    #     fp = open("out/corpus_onehot_text.txt","w",encoding="utf-8")
    #     for text_onehot in textL:
    #         text = [self.id2word(i.strip()) for i in text_onehot.split(",")]
    #         text_str = " ".join(text)
    #         fp.write(text_str+"\n")
    #     fp.close()

    def __len__(self):
        return len(self.word2id_dict)
    
    def __getitem__(self,word_id):
        if type(word_id) is not str:
            word_id = str(word_id)
        return self.id2word_dict[word_id]

    def inputs2str(self,inputs):
        ans_str = ""
        for i in inputs:
            if i in [2]:
                continue
            if i in [1,3]:
                break
            ans_str += self.id2word(i)
        return ans_str


if __name__ == '__main__':
    word_table = WordTable()
    word_table.load_dict()


    # print(f"word_table length: {len(word_table)}")

    # print(cal_similarity([1,2],[1,1]))