import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch

print(torch.cuda.is_available())

from datasets import Dataset

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.detector.generalized_vl_rcnn import GeneralizedVLRCNN
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from utils import build_transform, load_img

path = "../GLIP/"
config_file = path + "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = path + "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.BACKBONE.USE_CHECKPOINT", False])
cfg.merge_from_list(["MODEL.BACKBONE.FREEZE", True])

cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.merge_from_list(["MODEL.DEBUG", True])

transform = build_transform(cfg, min_image_size=800)


def data_from_arrow(path):
    print("loading data from folder:", path + '_arrow')
    return Dataset.load_from_disk(path + '_arrow', keep_in_memory=True)


@torch.no_grad()
def get_feature(image, model):
    image = transform(image)
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY).to(cfg.MODEL.DEVICE)
    return model(image_list)[0].to("cpu")


def prepare_data(data_path, model, json_path) -> Dataset:
    if os.path.exists(json_path + '_arrow'):
        return data_from_arrow(json_path)
    print("arrow not found, loading data from json:", json_path + '.json')
    if not os.path.exists(json_path + '.json'):
        raise ValueError("json path not exists:", json_path + '.json')

    '''
    example={
    "token": ["RT", "@Am_Blujay", ":", "Ronaldo", "trying", "to", "see", "if", "Messi", "is", "human", "#", "ElClasico"], 
    "img_id": "twitter_stream_2018_05_07_1_0_2_45.jpg", 
    "label_list": [
            [{"beg_ent": {"name": "Chiyori", "pos": [3, 4], "tags": "per"}, "sec_ent": {"name": "Miyu", "pos": [5, 6], "tags": "per"}, "relation": "peer"}], 
            [{"beg_ent": {"name": "Chiyori", "pos": [3, 4], "tags": "per"}, "sec_ent": {"name": "Miho", "pos": [7, 8], "tags": "per"}, "relation": "peer"}], 
            [{"beg_ent": {"name": "Miyu", "pos": [5, 6], "tags": "per"}, "sec_ent": {"name": "Miho", "pos": [7, 8], "tags": "per"}, "relation": "peer"}]
        ]
    }
    '''
    data = []
    with open(json_path + '.json', "r", encoding='utf-8') as fr:
        pieces = json.load(fr)
        # exit(-666)

        for one in tqdm(pieces):
            data_dict = {'img_id': one['img_id']}
            img = get_feature(load_img(data_path + '/' + one['img_id']), model).squeeze(0)
            # print(type(img), img.shape)
            data_dict['images'] = img
            data.append(data_dict)

        data = Dataset.from_list(data)

    print("saving data to arrow:", json_path + '_arrow')
    data.save_to_disk(json_path + '_arrow')

    return data


def data_folder(model, path="../few_shot"):
    train = prepare_data("../JMERE_text/train", model, path + "/train")
    print(train)
    val = prepare_data("../JMERE_text/val", model, path + "/val")
    print(val)
    test = prepare_data("../JMERE_text/test", model, path + "/test")
    print(test)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':

    glip: GeneralizedVLRCNN = build_detection_model(cfg)
    glip.eval()
    glip.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, glip, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    print("GLIP Loaded from:", cfg.MODEL.WEIGHT)

    debug = prepare_data("../JMERE_text/debug", glip, "../few_shot/debug")
    print(debug)

    for i, batch in enumerate(debug):
        print(i, batch['img_id'], [len(batch['images']),
                                   len(batch['images'][0]),
                                   len(batch['images'][0][0])])

    # exit()

    data_folder(glip, "../few_shot/seed_97")
    data_folder(glip, "../few_shot/seed_67")
    data_folder(glip, "../few_shot/seed_17")
