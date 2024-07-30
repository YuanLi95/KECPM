import os
import math

import torch
import numpy as np
import pickle as pkl
from src.utils import prompt_direct_inferring,cot_for_relation_inferring,cot_for_entities_inferring
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset,load_from_disk
import random
import json
import jsonlines
# from datasets import Dataset, disable_caching, DatasetBuilder, concatenate_datasets




class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_prompt(self, prompt_type, line,image_caption,knowledge):
        knowledge_setting = self.config.knowledge_number
        knowledge_number = min(knowledge_setting,len(knowledge))
        if prompt_type == "imagecap_know":
            knowledge_str = ""
            for idx in range(knowledge_number):
                knowledge_str += knowledge[idx] + ";"
            final_line = "<Img> image caption is: " + image_caption+" <\Img>"+ ". <Text> Text is: " + line+ "<\Text>" +"<Knowledge> the background knowledge: " + knowledge_str+" <\Knowledge>" \
                        + "."


        elif prompt_type == "imagecap":
            knowledge_str = ""
            for one_know in knowledge:
                knowledge_str += one_know
            final_line = "<Img> image caption is: " + image_caption+" <\Img>"+". <Text> Text is:" + line+" ."

        elif prompt_type == "know":
            knowledge_str = ""
            for one_know in knowledge:
                knowledge_str += one_know
            final_line = "<Knowledge> the background knowledge: " + knowledge_str+" <\Knowledge>"+ "<Text> Text is: " + line+" ."
        elif prompt_type == "only_text":
            final_line = "<Text> Text is: " + line+" ."
        else:
            assert  print("prompt_type is fault")
        final_line = ' '.join(final_line.split()[:self.config.max_length - 200])

        return final_line




    def get_data(self):
        cfg = self.config
        path =  '{}/{}/{}_{}.pkl'.format(cfg.data_dir, cfg.preprocessed_dir, cfg.model_size, cfg.model_path)

        # if os.path.exists(path):
        #     self.data = pkl.load(open(path, 'rb'))
        # else:

        train_data, valid_data, test_data = self.config.preprocessor.forward()
        # print(train_data)

        load_data_train = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=True, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)

        load_dat_test = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, shuffle=False, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)

        train_loader = load_data_train(train_data)
        # train_loader, valid_loader, test_loader = map(load_data_train, [train_data, valid_data, test_data])
        valid_loader, test_loader = map(load_dat_test,[valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        # print(*data)
        input_tokens, image_caption,knowledge,relation_list,label_list = zip(*data)

        if self.config.reasoning == 'prompt':

            contexts = []
            ent_rel_targets = []
            batch_image = []
            for i, line in enumerate(input_tokens):
                final_line = self.get_prompt(self.config.prompt_type, line, image_caption[i][0], knowledge[i])


                if self.config.zero_shot == True:
                    original_context, prompt,answer = prompt_direct_inferring(final_line, label_list[i])
                else:
                    original_context, prompt,answer = prompt_direct_inferring(final_line, label_list[i])
                contexts.append(prompt)
                ent_rel_targets.append(answer)
                # batch_image.append(image_feature)


            batch_input = self.tokenizer.batch_encode_plus(contexts, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            batch_output = self.tokenizer.batch_encode_plus(ent_rel_targets, max_length=self.config.max_length, padding=True,
                                                            return_tensors="pt").data


            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                # "image_feature" :batch_image["batch_image"]

            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':

            original_sentences = []
            contexts_relation = []
            relation_targets = []
            ent_rel_targets = []
            contexts_ent_rel = []
            fusion_all_input = []
            fusion_all_output = []

            for i, line in enumerate(input_tokens):

                final_line = self.get_prompt(self.config.prompt_type, line,image_caption[i][0],knowledge[i])

                original_sentence, context_relation,relation_tar = cot_for_relation_inferring(final_line,relation_list[i])

                context_ent_rel, ent_rel_tar = cot_for_entities_inferring(final_line, label_list[i])





                contexts_relation.append(context_relation)
                relation_targets.append(relation_tar)
                original_sentences.append(original_sentence)
                contexts_ent_rel.append(context_ent_rel)
                ent_rel_targets.append(ent_rel_tar)

                # fusion_all_input.append(context_relation)
                # fusion_all_input.append(context_ent_rel)
                # fusion_all_output.append(relation_targets)
                # fusion_all_output.append(ent_rel_targets)
            # print(original_sentences)


            batch_contexts_relation = self.tokenizer.batch_encode_plus(contexts_relation, padding="max_length", return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_relation =batch_contexts_relation.data
            batch_original_sentences = self.tokenizer.batch_encode_plus(original_sentences, padding="max_length", return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_original_sentences =batch_original_sentences.data
            batch_relation_targets = self.tokenizer.batch_encode_plus(relation_targets, padding="max_length", return_tensors='pt',
                                                                        max_length=self.config.max_length)
            batch_relation_targets =batch_relation_targets.data
            batch_ent_rel_targets = self.tokenizer.batch_encode_plus(ent_rel_targets, padding="max_length",return_tensors='pt',
                                                                      max_length=self.config.max_length)
            batch_ent_rel_targets =batch_ent_rel_targets.data
            batch_contexts_ent_rel = self.tokenizer.batch_encode_plus(contexts_ent_rel, padding="max_length", return_tensors='pt',
                                                                    max_length=self.config.max_length)
            batch_contexts_ent_rel =batch_contexts_ent_rel.data
            if self.config.model_type =="fusion":
                res = {
                    'input_ids' :torch.cat([batch_contexts_relation['input_ids'],batch_contexts_ent_rel['input_ids']],dim=0),
                    'input_masks':torch.cat([batch_contexts_relation['attention_mask'],batch_contexts_ent_rel['attention_mask']],dim=0),
                    'original_sentences_ids': batch_original_sentences['input_ids'],
                    'output_ids':torch.cat([batch_relation_targets['input_ids'],batch_ent_rel_targets['input_ids']],dim=0),
                    'output_masks':torch.cat([batch_relation_targets['attention_mask'],batch_ent_rel_targets['attention_mask']],dim=0),
                    'contexts_relation_ids': batch_contexts_relation['input_ids'],
                    'contexts_relation_masks': batch_contexts_relation['attention_mask'],
                    'relation_targets_ids': batch_relation_targets['input_ids'],
                    'relation_targets_masks': batch_relation_targets['attention_mask'],
                    'contexts_ent_rel_ids': batch_contexts_ent_rel["input_ids"],
                    'contexts_ent_rel_masks': batch_contexts_ent_rel["attention_mask"],
                    'ent_rel_targets_ids': batch_ent_rel_targets["input_ids"],
                    'ent_rel_targets_masks': batch_ent_rel_targets["attention_mask"]

                }
            else:
                res = {
                    'contexts_relation_ids': batch_contexts_relation['input_ids'],
                    'contexts_relation_masks': batch_contexts_relation['attention_mask'],
                    'original_sentences_ids': batch_original_sentences['input_ids'],
                    'relation_targets_ids': batch_relation_targets['input_ids'],
                    'relation_targets_masks': batch_relation_targets['attention_mask'],
                    'contexts_ent_rel_ids':batch_contexts_ent_rel["input_ids"],
                    'contexts_ent_rel_masks': batch_contexts_ent_rel["attention_mask"],
                    'ent_rel_targets_ids': batch_ent_rel_targets["input_ids"],
                    'ent_rel_targets_masks': batch_ent_rel_targets["attention_mask"]
                }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def data_from_arrow(self,path):
        print("loading data from folder:", path + '_arrow')

        # return Dataset.load_from_disk(path + '_arrow', keep_in_memory=True)
        return load_from_disk(path + '_arrow', keep_in_memory=True)

    def read_file(self):
        # path = "./few_shot/seed_17/train"
        # data = self.data_from_arrow(path)
        # print(data)
        # for i, j in data[0].items():
        #     print(i)
        # exit()
        train_file = os.path.join(self.config.data_dir, "train_ofa_knowledge.json")
        valid_file = os.path.join(self.config.data_dir, "val_ofa_knowledge.json")
        test_file =  os.path.join(self.config.data_dir, "test_ofa_knowledge.json")
        # print(train_file)
        # exit()
        train_data =  [line for line in jsonlines.Reader(open(train_file, 'r',encoding="utf-8"))]
        #
        valid_data = [line for line in jsonlines.Reader(open(valid_file, 'r',encoding="utf-8"))]
        #
        test_data =[line for line in jsonlines.Reader(open(test_file, 'r',encoding="utf-8"))]
        # train_data = self.data_from_arrow(os.path.join(self.config.data_dir, "train"))
        # valid_data = self.data_from_arrow(os.path.join(self.config.data_dir, "val"))
        # test_data = self.data_from_arrow(os.path.join(self.config.data_dir, "val"))
        # print(train_data)
        # ids = np.arange(len(train_data))
        # np.random.shuffle(ids)
        # np.random.seed(len(train_data))
        # np.random.shuffle(train_data)
        # print(train_data)

        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        res = []
        for i,line in enumerate(cur_data):


            # image_feature = line["images"]
            text = " ".join(line["token"])
            image_caption =   line["image_caption"]
            knowledge = line["knowledge"]
            label_list = line["label_list"]
            relation_list = []
            for label_triple in label_list:
                label_triple[0]['relation'] = label_triple[0]['relation'].replace("_", " ")
                new_relation_label = label_triple[0]['relation'].replace("_", " ")
                relation_list.append(new_relation_label)
            res.append([text,image_caption,knowledge,relation_list,label_list])
        return res

    def forward(self):
        # modes = 'train valid test'.split()
        train_data, valid_data, test_data = self.read_file()

        train_data = self.transformer2indices(train_data)
        print("----------------train_data_over---------------------------")
        valid_data = self.transformer2indices(valid_data)
        test_data  = self.transformer2indices(test_data )

        return train_data, valid_data, test_data
