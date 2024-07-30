

import regex as re
import jsonlines

relation_list = ["parent","siblings","couple" ,"neighbor","peer","charges","alumi","alternate names",
                 "place of residence","place of birth","member of", "subsidiary", "locate at","contain",
                 "present in","awarded","race","religion","nationality","part of", "held on"]

convert_to_idx = {"per":"person","loc":"location","org":"organization","misc":"miscellaneous"}

def polish_sentence(sentence):
    sentence = sentence.replace('_', ' ').replace('\n', '').replace(' #', '').strip()
        # replace("\"", "").replace("\'", "")
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
            # exit()
    return all_list



def get_score(predict,ground_truth_path):

    predict_list = get_list(predict)
    if predict_list==[]:
        p, r, f1 = 0,0,0
        return p,r,f1

    
    result_dict={}
    for key in relation_list:
        result_dict[key]={
            'num_correct' : 0,
            'num_ground' : 0,
            'num_pred' : 0,
        }

    with open(ground_truth_path, "r", encoding='utf-8') as fr:

        pieces = [line for line in jsonlines.Reader(fr)]
        for n in range(len(pieces)):
            output = pieces[n]['label_list']
            
            ground_dict = {}
            for key in relation_list:
                ground_dict[key]=[]
            for idx in range(len(output)):
                rel = output[idx][0]
                gt1 =  f"""The relation between {rel['beg_ent']['name']} ({convert_to_idx[rel['beg_ent']['tags']]}) and {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) is "{rel['relation']}" """
                gt1 = polish_sentence(gt1)
                gt2 = f"""The relation between {rel['sec_ent']['name']} ({convert_to_idx[rel['sec_ent']['tags']]}) and {rel['beg_ent']['name']}({convert_to_idx[rel['beg_ent']['tags']]}) is "{rel['relation']}" """
                # print(gt1)
                gt2 = polish_sentence(gt2)
                # print(gt2.split())
                ground_dict[rel['relation'].replace('_', ' ').strip()].extend([gt1, gt2])
            # print(ground_dict)
            
            # print("---------------000000000000------------------")
           
            pred_dict = {}
            for key in relation_list:
                pred_dict[key]=[]
            for pred_one in predict_list[n]:
                try:
                    # print(pred_one)
                    pred_rel=pred_one.split('"')[-2].strip()
                    # print(pred_rel,"--------------",pred_one)
                    pred_dict[pred_rel].extend([pred_one])
                except:
                    print("parsing relation failed !!! rel:","ans:",pred_one)
                    pass
            # print(pred_dict)

            # print("--------------------111111111111111-----------------")
           
            for key in relation_list:
                result_dict[key]['num_pred']+=len(pred_dict[key])
                result_dict[key]['num_ground']+=len(ground_dict[key])//2
                result_dict[key]['num_correct']+=len(set(pred_dict[key]) & set(ground_dict[key]))

            # break  # 1条数据

        for key in relation_list:
            num_correct=result_dict[key]['num_correct']
            num_pred=result_dict[key]['num_pred']
            num_ground=result_dict[key]['num_ground']
            p=num_correct/num_pred if num_pred!=0 else 0
            r=num_correct/num_ground if num_ground!=0 else 0
            result_dict[key]['p']=p
            result_dict[key]['r']=r
            result_dict[key]['f1']=2 * p * r / (p + r) if (p+r)!=0 else 0
        # print(result_dict)
        p,r,f1=0,0,0
        for key in relation_list:
            p+=result_dict[key]['p']
            r+=result_dict[key]['r']
            f1+=result_dict[key]['f1']
        p/=len(relation_list)
        r/=len(relation_list)
        f1/=len(relation_list)
        print(p,r,f1)
        return p,r,f1

if __name__  == "__main__":
    a = ['Answer: The relation between Perfect Day (miscellaneous) and Petaluma (location) is "held on"; The relation between RT @Ferrari (organization) and Ferrari (organization) is "subsidiary"']
    b = ['Answer: The relation between Ruby Rose (person) and Batwoman (miscellaneous) is "present in";','Answer: The relation between RT @Ferrari (organization) and Maranello (organization) is "subsidiary";The relation between RT @Ferrari (organization) and Ferrari (organization) is "alternate names";The relation between Ferrari (organization) and Maranello (organization) is "subsidiary";']
    # get_score(b,b)
    get_score(a,"../no_none_unified_tags_txt/seed_17/val_ofa_knowledge.json")

