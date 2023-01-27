from torch import nn
from transformers import T5ForConditionalGeneration


class FormalModule(nn.Module):
    def __init__(self, model_name):
        super(FormalModule, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids, attention_mask, labels):
        return self.t5_model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels
        )
