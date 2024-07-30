import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration,BartForConditionalGeneration


class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        if self.config.model_base == "T5":
            self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        elif self.config.model_base == "BART":
            self.engine = BartForConditionalGeneration.from_pretrained(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def generate(self, input_ids, input_masks):
        input_ids, input_masks = input_ids,input_masks
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, input_ids, input_masks):
        input_ids, input_masks = input_ids,input_masks
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=300)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output

    def test_to_word(self,input_ids):
        dec = [self.tokenizer.decode(ids) for ids in input_ids]

        output = [w.replace('<pad>', '').replace('</s>', '').strip() for w in dec]
        # print(output)
        return  output
