import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from src.utils import  cot_for_entities_test
from transformers import AutoTokenizer
import src.score as test_score
import src.score_macroF1  as score_macroF1
import copy
relation_list = ["parent","siblings","couple" ,"neighbor","peer","charges","alumi","alternate names",
                 "place of residence","place of birth","member of", "subsidiary", "locate at","contain","present in","awarded","race","religion","nationality","part of",
                 "held on"]


convert_to_idx = {"per":"person","loc":"location","org":"organization","misc":"miscellaneous"}



class PromptTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name =  config.save_name

        self.final_score = 0
        self.final_res = ''
        self.model_path = '{0}/savemodel/{1}'.format(self.config.data_dir,self.config.reasoning)

        self.scores, self.lines = [], []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            pre, recall, f1, mac_pre, mac_recall, mac_f1 = self.evaluate_step(dataLoader=self.valid_loader)
            score = f1

            if score > best_score:
                best_val_pre, best_val_recall, best_val_f1, best_mac_f1 = pre, recall, f1, mac_f1

                best_val_str = "this best val result:Pre:{0}--recall:{1}---F1:{2}-----mac_f1----{3}".format(
                    best_val_pre, best_val_recall, best_val_f1, best_mac_f1)
                print(best_val_str)

                best_score, best_iter,best_mac_f1  = score, epoch,mac_f1
                save_name = self.save_name + "epoch_{0}_best_score_{1}.pth.tar".format(epoch, best_score)

                if not os.path.exists(self.model_path):
                    os.makedirs(self.model_path)
                best_val_model = copy.deepcopy(self.model.cpu().state_dict(), )
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            print("this epoch val result:-----------Pre:{0}----------Recall:{1}--------F1:{2}------mac_f1:{3}".format(pre, recall, f1,
                                                                                                      mac_f1))
        torch.save({'epoch': epoch, 'model': best_val_model, 'best_score': best_score,"mac_f1":best_mac_f1},
                   save_name)

        self.model.to(self.config.device)
        self.model.to(self.config.device)

        best_val_str = "this best val result:Pre:{0}--Recall:{1}---F1:{2}".format(best_val_pre, best_val_recall,
                                                                                  best_val_f1)

        test_pre, test_recall, test_f1, mac_pre, mac_recall, mac_f1= self.final_evaluate(
            data_loader=self.test_loader, epoch=best_iter, best_score=best_score)

        test_str = "this test result: Pre:{0}-Recall:{1}--F1:{2}--\n --mac_pre:{3}--mac_recall{4}, mac_f1:{5}".format(test_pre, test_recall, test_f1,
                                                                                     mac_pre, mac_recall, mac_f1)


        print(test_str)
        return best_val_str, test_str

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        print("11111111111111")
        losses = []
        for i, data in enumerate(train_data):

            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        if dataLoader == self.valid_loader:
            ground_truth_path = os.path.join(self.config.data_dir, "val_ofa_knowledge.json")

        else:
            ground_truth_path = os.path.join(self.config.data_dir, "test_ofa_knowledge.json")
        # print(ground_truth_path)
        # exit()

        all_predict = []
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                # print(data)
                output = self.model.generate(input_ids=data["input_ids"],input_masks=data["input_masks"])
                # print(output)
                # print("this is output")
                # exit()
                all_predict.extend(output)
            print("-----------------------{0}-------------------".format(i))
            print(output)
            # print()
            print(self.model.test_to_word(data["output_ids"]))

        pre, recall, f1 = test_score.get_score(all_predict, ground_truth_path)
        mac_pre, mac_recall, mac_f1 = score_macroF1.get_score(all_predict, ground_truth_path)
        return pre, recall, f1, mac_pre, mac_recall, mac_f1



    def final_evaluate(self,data_loader=None,epoch=0,best_score=0):
        data_loader = data_loader
        PATH =   self.save_name+"epoch_{0}_best_score_{1}.pth.tar".format(epoch,best_score)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        pre,recall,f1,mac_pre, mac_recall, mac_f1 = self.evaluate_step(data_loader)
        return pre,recall,f1,mac_pre, mac_recall, mac_f1



    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits']





class ThorTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name =  config.save_name
        # self.save_name =config.save_name
        print(self.save_name)
        self.final_score = 0
        self.final_res = ''
        self.scores, self.lines = [], []
        self.re_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        self.model_path = '{0}/savemodel/{1}_{2}'.format(self.config.data_dir,self.config.reasoning,self.separate)

    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            print("---------------------this is epoch {0}---------------".format(epoch))
            pre,recall, f1,mac_pre,mac_recall, mac_f1 = self.evaluate_step(self.valid_loader)
            print( "this epoch val result:Pre:{0}--recall:{1}---Recall:{2}-----mac_f1----------{3}".format(pre, recall, f1,mac_f1))

            score =  f1

            if score > best_score:
                best_val_pre,best_val_recall,best_val_f1,best_mac_f1 =pre,recall, f1,mac_f1

                best_val_str = "this best val result:Pre:{0}--recall:{1}---F1:{2}-----mac_f1----{3}".format(best_val_pre,best_val_recall,best_val_f1,best_mac_f1)
                print(best_val_str)
                best_score, best_iter = score, epoch

                save_name = self.save_name+"epoch_{0}_best_score_{1}.pth.tar".format(epoch,best_score)

                if not os.path.exists(self.model_path):
                    os.makedirs(self.model_path )
                best_val_model = copy.deepcopy(self.model.cpu().state_dict(),)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
        torch.save({'epoch': best_iter, 'model': best_val_model, 'best_score': best_score,"mac_f1":best_mac_f1 },
                   save_name,)

        self.model.to(self.config.device)

        best_val_str = "this best val result:Pre:{0}--Recall:{1}---F1:{2}".format(best_val_pre,best_val_recall,best_val_f1)


        test_pre,test_recall, test_f1 ,test_mac_f1 = self.final_evaluate(data_loader=self.test_loader,epoch=best_iter,best_score=best_score)


        test_str = "this test result: Pre:{0}-Recall:{1}--F1:{2}---MacF1:{3}".format(test_pre,test_recall, test_f1,test_mac_f1 )
        print(test_str)
        return  best_val_str,test_str



    def prepare_step_two(self, aspect_exprs, data):
        # print(aspect_exprs)
        ans_list = []
        for relation_one in aspect_exprs:
            print(relation_one)
            _, ans =relation_one.split(":",1)
            new_ans = []
            for item in ans.split(";"):
                item = item.strip()
                if item in relation_list:
                    new_ans.append(item)
            ans_list.append(new_ans)

        original_context_ids = data["original_sentences_ids"]
        original_context = [self.model.tokenizer.decode(ids) for ids in original_context_ids]
        original_context = [context.replace('<pad>', '').replace('</s>', '').strip() for context in original_context]


        # exit()

        result_prompts = []

        for context, ans_relation in zip(original_context,ans_list):
            # print(context)
            # print(type(context))
            # print(type(ans_relation))
            prompt = cot_for_entities_test(context, ans_relation)
            result_prompts.append(prompt)

        # print(result_prompts)

        # exit()
        batch_inputs = self.model.tokenizer.batch_encode_plus(result_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data



        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res



    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length)

        losses = []
        for i, data in enumerate(train_data):
            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)

            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        if dataLoader == self.valid_loader:
            # print("111111111111111111111111111111")
            ground_truth_path =  os.path.join(self.config.data_dir, "val_ofa_knowledge.json")

        else:
            ground_truth_path = os.path.join(self.config.data_dir, "test_ofa_knowledge.json")

        dataiter = dataLoader
        # print(111111111111)
        # exit()
        all_predict= []


        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                contexts_relation_ids = data["contexts_relation_ids"]
                # print(self.model.test_to_word(data["contexts_relation_ids"]))
                # exit()
                contexts_relation_masks = data["contexts_relation_masks"]
                step_one_inferred_output = self.model.generate(input_ids=contexts_relation_ids,input_masks=contexts_relation_masks)

                step_two_inferred_data = self.prepare_step_two(step_one_inferred_output, data)


                step_two_inferred_output = self.model.generate(step_two_inferred_data["input_ids"],step_two_inferred_data["input_masks"])


            all_predict.extend(step_two_inferred_output)

            print("------------------------------------------")
            print(step_two_inferred_output)
            print(self.model.test_to_word(data["ent_rel_targets_ids"]))


        pre,recall, f1 = test_score.get_score(all_predict,ground_truth_path)
        mac_pre,mac_recall, mac_f1 = score_macroF1.get_score(all_predict,ground_truth_path)

        return pre,recall, f1,mac_pre,mac_recall, mac_f1

    def final_evaluate(self,data_loader=None,epoch=0,best_score=0):
        data_loader = data_loader
        PATH =   self.save_name+"epoch_{0}_best_score_{1}.pth.tar".format(epoch,best_score)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        pre,recall,f1,mac_f1 = self.evaluate_step(data_loader)
        return pre,recall,f1,mac_f1

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits']

