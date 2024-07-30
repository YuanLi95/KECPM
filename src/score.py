

import regex as re
import jsonlines
num_correct = 0
num_ground = 0
num_pred = 0

relation_list = ["parent","siblings","couple" ,"neighbor","peer","charges","alumi","alternate names",
                 "place of residence","place of birth","member of", "subsidiary", "locate at","contain","present in","awarded","race","religion","nationality","part of",
                 "held on"]

convert_to_idx = {"per":"person","loc":"location","org":"organization","misc":"miscellaneous"}

def polish_sentence(sentence):
    sentence = sentence.replace('_', ' ').replace('\n', '').replace(' #', '').\
        replace("\"","").replace("\'","").strip()
    return sentence

def get_list(text_list):
    all_list = []
    for index, item in enumerate(text_list):
        # print(item)
        if "Answer" in item:

            item = item.replace("<s>", "")
            item = item.replace("Answer: ", "")
            item = item.replace("Answer (Text): ", "")

            # print(item)
            tmp = item.split(";")


            tmp = [polish_sentence(one) for one in tmp if one != ""]
            # print(tmp)
            all_list.append(set(tmp))
        else:
            print(item)
            exit()
    return all_list



def get_score(predict,ground_truth_path):

    predict_list = get_list(predict)
    # print(predict_list)
    # exit()
    if predict_list==[]:
        p, r, f1 = 0,0,0
        return p,r,f1

    num_correct = 0
    num_ground = 0
    num_pred = 0
    with open(ground_truth_path, "r", encoding='utf-8') as fr:

        pieces = [line for line in jsonlines.Reader(fr)]
        # print(len(pieces))
        # print(len(predict_list))
        # exit()
        for n in range(len(pieces)):
            output = pieces[n]['label_list']
            ground = []
            for idx in range(len(output)):

                rel = output[idx][0]


                gt1 =  f"""The relation between {rel['beg_ent']['name']} ({convert_to_idx[rel['beg_ent']['tags']]}) and {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) is "{rel['relation']}"""
                gt1 = polish_sentence(gt1)
                gt2 = f"""The relation between {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) and {rel['beg_ent']['name']}({convert_to_idx[rel['beg_ent']['tags']]}) is "{rel['relation']}"""
                # print(gt1)
                gt2 = polish_sentence(gt2)
                #

                num_ground += 1
                ground.extend([gt1, gt2])
            # print("1-----------------------------------------------------------------")
            # print(pieces[n])
            print(ground)
            print(predict_list[n])
            # print("---------")
            # exit()
            ground = set(ground)
            num_pred += len(predict_list[n])
            num_correct += len(ground & predict_list[n])
            # print(ground - predict_list[n], "---------", predict_list[n] - ground)


        print(num_correct, num_pred, num_ground)
        if num_correct==0:
            p, r, f1=0,0,0
        else:
            p = num_correct / num_pred
            r = num_correct / num_ground
            f1 = 2 * p * r / (p + r)
        print(p,r,f1)
        return p,r,f1

if __name__  == "__main__":
    a = ['Answer: The relation between Perfect Day (miscellaneous) and Petaluma (location) is "held on"; The relation between RT @Ferrari (organization) and Ferrari (organization) is "subsidiary"']
    b = ['Answer: The relation between Ruby Rose (person) and Batwoman (miscellaneous) is "present in";','Answer: The relation between RT @Ferrari (organization) and Maranello (organization) is "subsidiary";The relation between RT @Ferrari (organization) and Ferrari (organization) is "alternate names";The relation between Ferrari (organization) and Maranello (organization) is "subsidiary";']
    # get_score(b,b)
    get_score(a,"../no_none_unified_tags_txt/seed_17/val_ofa_knowledge.json")

