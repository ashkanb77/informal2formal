import pandas as pd


def read_dataset(train_informal_path, val_informal_path, dict_path, min_count):

    train_df = pd.read_csv(train_informal_path)
    train_df = train_df[['inFormalForm', 'FormalForm']]
    train_df.dropna(inplace=True)

    val_df = pd.read_csv(val_informal_path)
    val_df = val_df[['inFormalForm', 'FormalForm']]
    val_df.dropna(inplace=True)

    dictionary = read_dict(dict_path, min_count)

    return dictionary, train_df, val_df


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