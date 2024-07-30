import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup



convert_to_idx = {"per":"person","loc":"location","org":"organization","misc":"miscellaneous"}



def prompt_direct_inferring(context,entities_list):
    original_context = context
    new_context = f'Given a list of relations: [parent, siblings, couple, neighbor, peer, charges, alumi, alternate names, place of residence, place of birth, member of, subsidiary, locate at, contain, present in, awarded, race, religion, nationality, part of, held on], Given a list of types: [person, location, organization, miscellaneous]. Given a Sentence "{context}",'

    prompt = new_context + f'Describes entities with corresponding types and their relations in the sentence.'
    answer = "Answer: "
    for rel in entities_list:
        rel = rel[0]
        answer += f"""The relation between {rel['beg_ent']['name']} ({convert_to_idx[rel['beg_ent']['tags']]}) and {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) is "{rel['relation']}";"""

    return original_context, prompt,answer



def cot_for_relation_inferring(context,relation_target):
    new_context = f'Given a list of relations: [parent, siblings, couple, neighbor, peer, charges, alumi, alternate names, place of residence, place of birth, member of, subsidiary, locate at, contain, present in, awarded, race, religion, nationality, part of, held on]. Given a Sentence "{context}",'
    original_context = context
    prompt = new_context + f'which relations are possibly mentioned?'
    answer = "Answer: "
    ##relation_target she ji
    for rel in relation_target:
        answer+=f'{rel};'
    # print(prompt)
    # print(answer)
    return original_context, prompt,answer




def cot_for_entities_inferring(context,entities_list):
    relation_sentence  = "["
    for i,item in enumerate(entities_list):
        item = item[0]
        relation = item["relation"]
        if i<len(entities_list)-1:
            relation_sentence += f'{relation}'+","
        else:
            relation_sentence += f'{relation}'
    relation_sentence+="]"
    # new_context = f'Given a list of types: [person, location, organization, miscellaneous]. Given a Sentence "{context}", with contained relations list "{relation_sentence}",'
    new_context = f'Given a list of relations: [parent, siblings, couple, neighbor, peer, charges, alumi, alternate names, place of residence, place of birth, member of, subsidiary, locate at, contain, present in, awarded, race, religion, nationality, part of, held on], Given a list of types: [person, location, organization, miscellaneous]. Given a Sentence "{context}", with contained relations list "{relation_sentence}",'
    prompt = new_context + f'Describes entities with corresponding types and their relations in the sentence.'
    answer = "Answer: "
    for rel in entities_list:
        rel = rel[0]
        answer+=f"""The relation between {rel['beg_ent']['name']} ({convert_to_idx[rel['beg_ent']['tags']]}) and {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) is "{rel['relation']}";"""
    # print("--------------------------------------------cot_entities_inferring---------------------------------------------")
    return  prompt,answer

def cot_for_entities_test(context,relation_list):
    relation_sentence = "["
    for i, item in enumerate(relation_list):
        if i<len(relation_list)-1:
            relation_sentence += f'{item}'+","
        else:
            relation_sentence += f'{item}'
    relation_sentence+="]"
    new_context = f'Given a list of types: [person, location, organization, miscellaneous]. Given a Sentence "{context}", with contained relations list "{relation_sentence}", '
    prompt = new_context + f'Describes entities with corresponding types and their relations in the sentence.'
    return prompt

def cot_for_entities_result(context,relation_list):
    return 0

def prompt_inferring_all(context):
    return 0



def cot_for_entities_test(context,relation_list):
    # print(type(relation_list))
    relation_sentence = "["
    for i, item in enumerate(relation_list):
        # print(item)
        if i<len(relation_list)-1:
            relation_sentence += f'{item}'+","
        else:
            relation_sentence += f'{item}'
    relation_sentence+="]"
    # print(relation_sentence)
    # new_context = f'Given a list of types: [person, location, organization, miscellaneous]. Given a Sentence "{context}", with contained relations list "{relation_sentence}", '
    new_context = f'Given a list of relations: [parent, siblings, couple, neighbor, peer, charges, alumi, alternate names, place of residence, place of birth, member of, subsidiary, locate at, contain, present in, awarded, race, religion, nationality, part of, held on], Given a list of types: [person, location, organization, miscellaneous].Given a Sentence "{context}", with contained relations list "{relation_sentence}",'

    prompt = new_context + f'Describes entities with corresponding types and their relations in the sentence.'
    return prompt



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res
