import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(train_informal_path, dict_path, min_count, split=True):
    df = pd.read_csv(train_informal_path)
    df.rename(columns={"formalForm": "FormalForm"}, inplace=True)
    df = df[['inFormalForm', 'FormalForm']]

    df.dropna(inplace=True)
    dictionary = read_dict(dict_path, min_count)

    if split:
        train_df, val_df = train_test_split(df, test_size=0.1)
        return dictionary, train_df, val_df

    return dictionary, df


def read_dict(dict_path, min_count):
    dic_df = pd.read_csv(dict_path)
    dic_df.dropna(inplace=True)
    formal_words = set(dic_df.formal.values)
    dic_df = dic_df[dic_df['count'] >= min_count]

    dictionary = dict()

    for key, value in zip(dic_df.informal, dic_df.formal):
        v = dictionary.get(key)
        if not v:
            dictionary[key] = [value]
        else:
            dictionary[key] = dictionary[key] + [value]

        if key in formal_words:
            dictionary[key] += [key]
    return dictionary


def find_formal_forms(sentences, dic):
    res = []
    for sentence in sentences:
        s = []
        for k in sentence.split(' '):
            v = dic.get(k)
            if v:
                s = s + v

        res.append(sentence + ' ' + ' '.join(s))
    return res


def collate_fn(data, tokenizer):
    formal, informal = zip(*data)
    formal = list(formal)
    informal = list(informal)

    tokenized = tokenizer(informal, text_target=formal, padding=True, return_tensors='pt')
    return tokenized
