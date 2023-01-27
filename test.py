import torch
from argparse import ArgumentParser
from model import FormalModule
from transformers import AutoTokenizer
from utils import find_formal_forms, read_dict


parser = ArgumentParser()

parser.add_argument('--tokenizer', type=str, default='erfan226/persian-t5-paraphraser', help='tokenizer')
parser.add_argument('--model', type=str, default='erfan226/persian-t5-paraphraser', help='model')
parser.add_argument('--model_checkpoint', type=str, default='model.pth', help='dataset directory')
parser.add_argument('--dict_path', type=str, default='dataset/dict.csv', help='dictionary path')
parser.add_argument('--min_count', type=int, default=10, help='min count ')
parser.add_argument('--informal_texts', type=str, default='informals.txt')
parser.add_argument('--append_formals', type=bool, default=False,
                    help='append formal forms to end of informal sentence')

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FormalModule(args.model)
model.load_state_dict(torch.load(args.model_checkpoint))
model.to(device)


def convert(text):
    text_encoding = tokenizer(
      text,
      padding=True,
      return_tensors='pt'
    )
    text_encoding = {k: v.to(device) for k, v in text_encoding.items()}

    generated_ids = model.t5_model.generate(
      input_ids=text_encoding['input_ids'],
      attention_mask=text_encoding['attention_mask'],
      early_stopping=True
    )

    preds = [
           tokenizer.decode(gen_id, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True) for gen_id in generated_ids
    ]

    return "".join(preds)


with open(args.informal_texts, 'r') as file:
    sentences = []
    for line in file:
        sentences.append(line)


dic = read_dict(args.dict_path, args.min_count)

if args.append_formals:
    sentences = find_formal_forms(sentences, dic)

res = []
for sentence in sentences:
    res.append(convert(sentence))

for r in res:
    print(r)
