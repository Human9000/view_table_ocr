import os
import pandas as pd

head_01 = [['0食物编码/Food code',
            '0食物名称/Food name',
            '烟酸/Niacin/mg',
            '维生素C/Vitamin C/mg',
            '维生素E/Vitamin E/Total/mg',
            '维生素E/Vitamin E/α-E/mg',
            '维生素E/Vitamin E/(β+γ)-E/mg',
            '维生素E/Vitamin E/σ-E/mg',
            '钙/Ca/mg',
            '磷/P/mg',
            '钾/K/mg',
            '钠/Na/mg',
            '镁/Mg/mg',
            '铁/Fe/mg',
            '锌/Zn/mg',
            '硒/Se/μg',
            '铜/Cu/mg',
            '锰/Mn/mg',
            '备注/Remark'],
           ['1食物编码/Food code',
            '1食物名称/Food name',
            '食部/Edible/%',
            '水/Water/g',
            '能量/Energy/kcal',
            '能量/Emergy/kJ',
            '蛋白质/Protein/g',
            '脂肪/Fat/g',
            '碳水化合物/CHO/g',
            '膳食纤维/Dietary fiber/g',
            '胆固醇/Cholesterol/mg',
            '灰分/Ash/g',
            '维生素A/Vitamin A/μgRAE',
            '胡萝卜素/Carotene/μg',
            '视黄醇/Retinol/μg',
            '硫胺素/Thiamin/mg',
            '核黄素/Riboflavin/mg'], ]

head_child_01 = [['0食物编码/Food code',
                  '0食物名称/Food name',
                  '维生素K/Vitamin K/μg',
                  '维生素B6/Vitamin B6/mg',
                  '维生素B12/Vitamin B12/μg',
                  '叶酸/Vitamin K/μg',
                  '生物素/Vitamin K/μg',
                  '泛酸/Vitamin K/mg',
                  '胆碱/Vitamin K/mg',
                  '钙/Ca/mg',
                  '磷/P/mg',
                  '钾/K/mg',
                  '钠/Na/mg',
                  '镁/Mg/mg',
                  '铁/Fe/mg',
                  '锌/Zn/mg',
                  '硒/Se/μg',
                  '铜/Cu/mg',
                  '锰/Mn/mg',
                  '碘/I/μg',
                  '备注/Remark'],
                 ['1食物编码/Food code',
                  '1食物名称/Food name',
                  '食部/Edible/%',
                  '水/Water/g',
                  '能量/Energy/kcal',
                  '能量/Emergy/kJ',
                  '蛋白质/Protein/g',
                  '脂肪/Fat/g',
                  '碳水化合物/CHO/g',
                  '膳食纤维/Dietary fiber/g',
                  '胆固醇/Cholesterol/mg',
                  '灰分/Ash/g',
                  '维生素A/Vitamin A/μgRAE',
                  '胡萝卜素/Carotene/μg',
                  '硫胺素/Thiamin/mg',
                  '核黄素/Riboflavin/mg',
                  '烟酸/Niacin/mg',
                  '维生素C/Vitamin C/mg',
                  '维生素D/Vitamin E/mg',
                  # '维生素E/α-TE/mg',
                  '维生素E/Vitamin E/Total/mg',
                  '维生素E/Vitamin E/α-E/mg',
                  # '维生素E/α-TE/mg',
                  ], ]
pages_2 = [75, 95, 105, 135, 141, 167, 199, 203]
groups_2 = ["畜肉类及制品",
            "禽肉类及制品",
            "乳类及制品",
            "蛋类及制品",
            "鱼虾蟹贝类",
            "婴幼儿食品",
            "油脂类(动物)",
            "其他",
            ]
pages_1 = [52, 66, 72, 82, 112, 110, 130, 148]
groups_1 = ["谷类及制品",
            "薯类、淀粉及制品",
            "干豆类及制品",
            "蔬菜类及制品",
            "菌藻类",
            "水果类及制品",
            "坚果、种子类",
            "油脂类（植物）",
            ]
# new_head = head_01[0][:2] + head_01[1][:2] + head_01[0][2:] + head_01[1][2:] + ['组别/Group']

new_head = ['0食物编码/Food code', '0食物名称/Food name', '1食物编码/Food code', '1食物名称/Food name', '食部/Edible/%', '水/Water/g',
            '能量/Energy/kcal', '能量/Emergy/kJ', '蛋白质/Protein/g', '脂肪/Fat/g', '碳水化合物/CHO/g', '膳食纤维/Dietary fiber/g',
            '胆固醇/Cholesterol/mg', '灰分/Ash/g', '维生素A/Vitamin A/μgRAE', '胡萝卜素/Carotene/μg', '视黄醇/Retinol/μg',
            '硫胺素/Thiamin/mg', '核黄素/Riboflavin/mg', '烟酸/Niacin/mg', '维生素C/Vitamin C/mg', '维生素D/Vitamin E/mg',
            '维生素E/Vitamin E/Total/mg', '维生素E/Vitamin E/α-E/mg', '维生素E/Vitamin E/(β+γ)-E/mg', '维生素E/Vitamin E/σ-E/mg',
            '维生素K/Vitamin K/μg', "维生素B6/Vitamin B6/mg", "维生素B12/Vitamin B12/μg", '叶酸/Vitamin K/μg', '生物素/Vitamin K/μg',
            '泛酸/Vitamin K/mg', '胆碱/Vitamin K/mg', '钙/Ca/mg', '磷/P/mg', '钾/K/mg', '钠/Na/mg', '镁/Mg/mg', '铁/Fe/mg',
            '锌/Zn/mg', '硒/Se/μg', '铜/Cu/mg', '锰/Mn/mg', '碘/I/μg', '备注/Remark', '组别/Group']

df_new = pd.DataFrame(columns=new_head)

for root, groups, pages, first01 in [
    ('o1', groups_1, pages_1, 0),
    ('o2', groups_2, pages_2, 1),
]:
    for name in os.listdir(root):
        path = f"{root}/{name}"
        page = int(name.split('.')[0])
        if page % 2 == first01:
            continue
        group = ''
        for p, g in zip(pages, groups):
            if page >= p:
                group = g
            else:
                break
        print(path, page, group)
        df1 = pd.read_csv(path, sep='\t', header=0, dtype=str)
        df1 = df1.fillna('')
        # print(df1)
        df1 = df1.drop(df1[~df1['0'].str.match(r'.*?[a-z0-9]+')].index)
        df1['1'] = df1["1"].str.replace('】', ']')
        df1['1'] = df1["1"].str.replace('【', '[')

        path = f"{root}/{page + 1}.tsv"
        df2 = pd.read_csv(path, sep='\t', header=0, dtype=str)
        df2 = df2.fillna('')
        df2 = df2.drop(df2[~df2['0'].str.match(r'.*?[a-z0-9]+')].index)
        df2['1'] = df2["1"].str.replace('】', ']')
        df2['1'] = df2["1"].str.replace('【', '[')

        if group == '婴幼儿食品':
            df1['20'] = df1['19']
            df1.columns = head_child_01[1]
            df2.columns = head_child_01[0]
        else:
            df1.columns = head_01[1]
            df2.columns = head_01[0]

        if df1.shape[0] != df2.shape[0]:
            print('error')
            print(df1, df2)
            exit(0)

        df = pd.concat([df1, df2], axis=1)
        df.insert(0, '组别/Group', group)
        for i in new_head:
            if i not in df.columns:
                df.insert(0, i, '')


        # df = df[new_head]
        # print(df1)
        # print(df2)
        df_new = pd.concat([df_new, df], axis=0)
        # break
print(df_new)
# df_new.sort_index(axis=0)
print(df_new)
df_new.to_excel(f"res.xlsx", index=False)
