from torch.utils.data import Dataset
from utils import find_formal_forms


class FormalDataset(Dataset):
    def __init__(self, dic, df, append_formals=True):
        self.dic = dic
        self.df = df
        if append_formals:
            self.df['inFormalForm'] = find_formal_forms(df['inFormalForm'], dic)

    def __getitem__(self, item):
        return (
            self.df.FormalForm[item],
            self.df.inFormalForm[item]
        )

    def __len__(self):
        return len(self.df)


