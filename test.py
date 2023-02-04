import torch
from argparse import ArgumentParser
from model import FormalModule
from transformers import AutoTokenizer
from utils import find_formal_forms, read_dataset
from datasets import load_metric
from dataset import FormalDataset
from torch.utils.data import DataLoader
from utils import collate_fn
import pandas as pd
from rouge import Rouge
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument('--tokenizer', type=str, default='erfan226/persian-t5-paraphraser', help='tokenizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='erfan226/persian-t5-paraphraser', help='model')
parser.add_argument('--model_checkpoint', type=str, default='model.pth', help='dataset directory')
parser.add_argument('--dict_path', type=str, default='dataset/dict.csv', help='dictionary path')
parser.add_argument('--min_count', type=int, default=10, help='min count ')
parser.add_argument('--validation_path', type=str, default='val_informals.csv')
parser.add_argument('--append_formals', type=bool, default=False,
                    help='append formal forms to end of informal sentence')

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FormalModule(args.model)
model.load_state_dict(torch.load(args.model_checkpoint))
model.to(device)

rouge = Rouge()


def convert_compute(input_ids, attention_mask, labels):
    generated_ids = model.t5_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, max_new_tokens='longest'
    )

    pred_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

    rouge_output = rouge.get_scores(pred_str, label_str)
    for row in rouge_output:
        scores['rouge-1'] += row['rouge-1']['f']
        scores['rouge-2'] += row['rouge-2']['f']
        scores['rouge-l'] += row['rouge-l']['f']

    scores = {k: v / len(rouge_output) for k, v in scores.items()}

    return input_str, pred_str, label_str, scores
    # return {['rouge2'].mid
    #     "rouge2_precision": round(rouge_output.precision, 4),
    #     "rouge2_recall": round(rouge_output.recall, 4),
    #     "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    # }


dic, val_df = read_dataset(args.validation_path, args.dict_path, args.min_count, split=False)

val_dataset = FormalDataset(dic, val_df, args.append_formals)
val_dataloader = DataLoader(val_dataset, collate_fn=lambda data: collate_fn(data, tokenizer),
                            batch_size=args.batch_size)

d = {'informals': [], 'prediction_formals': [], 'ground_truth_formals': []}
scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
i = 0
with tqdm(val_dataloader, unit="batch") as tepoch:
    for batch in tepoch:
        tepoch.set_description(f"Validation")

        batch = {k: v.to(device) for k, v in batch.items()}
        inputs_str, preds_str, labels_str, rouge_output = convert_compute(**batch)

        scores = {k: scores[k] + v for k, v in rouge_output.items()}
        i += 1

        d['informals'] = d['informals'] + inputs_str
        d['prediction_formals'] = d['prediction_formals'] + preds_str
        d['ground_truth_formals'] = d['ground_truth_formals'] + labels_str
        tepoch.set_postfix(rouge1=rouge_output['rouge-1'], rouge2=rouge_output['rouge-2'],
                           rougel=rouge_output['rouge-l'])

scores = {k: v / i for k, v in scores.items()}
print(scores)
df = pd.DataFrame(d)
df.to_csv('validation_path.csv')