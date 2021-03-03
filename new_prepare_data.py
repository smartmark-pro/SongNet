import collections
import sys
import re
from collections import Counter
import random
import pandas as pd
random.seed(10)

example_df = pd.read_excel(
    "./data/new_examples.xlsx")
# /content/drive/MyDrive/examples.xlsx
# ./data/new_examples.xlsx

print(example_df.shape)

data = example_df.to_dict("records")

category_data = collections.defaultdict(list)


def add_split_tag(text, ftext):
    # 暂时根据文字加标点吧, 后面考虑根据原始加?
    new_text = []
    new_ftext = []
    c = 0

    # if len(text) == 0 or len(ftext) == 0:
    #     print(text, ftext)
    #     return "".join(new_text), "".join(new_ftext), c
    for i in range(len(text)):
        if text[i] in set(["？", "！", "，", "。"]):
            new_text.append(text[i] + "</s>")
            new_ftext.append(ftext[i] + "</s>")
            c += 1
        elif text[i] == "\n":
            # 必须去掉换行, 否则放到txt中无法使用
            continue
        else:
            new_text.append(text[i])
            new_ftext.append(ftext[i])

    return "".join(new_text), "".join(new_ftext), c


c = 0
for i, item in enumerate(data):
    # 需要把每一句对上加一个</s>
    # 把标点直接替换成
    new_text, new_format, c2 = add_split_tag(
        str(item["标准化内容"]), str(item["标准化模板"]))

    text = ("{}<s1>{}<s2>{}".format(
        str(item["仿写对象"]).replace("\n", ""), str(item["梗和主题"]).replace("\n", ""), new_format), new_text)
    if len(new_format) != len(new_text):
        print(i, c, c2, len(new_format), len(new_text),
              new_format[:100], new_text[:100])
        c += 1

    category_data[item["梗和主题"]].append(text)

# 类别数量, 有没有错误结果
print(len(category_data), c)

all, tc = 0, 0
new_category_data = collections.defaultdict(list)
for k, v in category_data.items():
    if len(v) >= 4:  # 一个梗至少有四个结果
        print(k, len(v), end="\t")
        tc += 1
        all += len(v)
        new_category_data[k] = v

# 梗数量, 条数
print(tc, all)

# 暂时去掉吧
# 1100 选了50
# 8, 1,1
objects = list(category_data.keys())
random.shuffle(objects)
# 8:1:1
train_obj = objects[:int(0.8*len(objects))]
dev_obj = objects[int(0.8*len(objects)):int(0.9*len(objects))]
test_obj = objects[int(0.9*len(objects)):]


def save_data(name, data):
    with open("./data/r_{}.txt".format(name), "w") as fw:
        for a, b in data:
            fw.write("{}\t{}\n".format(a, b))


def get_data(keys, name):
    data = []
    for k in keys:
        data.extend(category_data[k])
    # 类别, 数量, 取得是先混合后的
    # 可以考虑按类别划分
    print(len(keys), len(data))
    save_data(name, data)
    return data


train_data = get_data(train_obj, "train")
dev_data = get_data(dev_obj, "dev")
test_data = get_data(test_obj, "test")
