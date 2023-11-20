import os
import pandas as pd

df_new = pd.DataFrame()

for root,  first01 in [
    ('o4',   1),
]:
    for name in os.listdir(root):
        path = f"{root}/{name}"
        page = int(name.split('.')[0])
        df1 = pd.read_csv(path, sep='\t', header=0, dtype=str)
        df1 = df1.fillna('')
        df1 = df1.drop(df1[df1['2']+df1['3']+df1['4']+df1['5']+df1['6']==''].index)
        df1['1'] = df1["1"].str.replace('】', ']')
        df1['1'] = df1["1"].str.replace('【', '[')
        df_new = pd.concat([df_new, df1], axis=0)

df_new.columns = ['序号/code', '食物名称/Food name', '鸟嘌呤/Guanine', '腺嘌呤/Adenine', '次黄嘌呤/Hypoxanthine', '黄嘌呤/Xanthine', '总嘌呤含量/Purine', '采样地/Sampling site']
print(df_new)
df_new.to_excel(f"嘌呤.xlsx", index=False)
