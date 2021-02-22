import collections
import sys
import re
from collections import Counter
import random
import pandas as pd
random.seed(10)

example_df = pd.read_excel("./data/examples.xlsx")

print(example_df.shape)

data = example_df.to_dict("records")

category_data = collections.defaultdict(list)


def add_split_tag(text):
    new = ""
    for i in text:
        if i in set(["？", "！", "，", "。"]):
            new += i + "</s>"
        elif i == "\n":
            continue
        else:
            new += i
    return new


for item in data:
    # 需要把每一句对上加一个</s>
    # 把标点直接替换成
    new_format = add_split_tag(item["标准化模板"])
    new_text = add_split_tag(item["文本内容"])
    text = ("{}<s1>{}<s2>{}".format(
        item["仿写对象"], item["梗和主题"], new_format), new_text)

    category_data[item["梗和主题"]].append(text)

print(len(category_data))

all, tc = 0, 0
new_category_data = collections.defaultdict(list)
for k, v in category_data.items():
    if len(v) >= 4:
        print(k, len(v), end="\t")
        tc += 1
        all += len(v)
        new_category_data[k] = v

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
    print(len(keys), len(data))
    save_data(name, data)
    return data


train_data = get_data(train_obj, "train")
dev_data = get_data(dev_obj, "dev")
test_data = get_data(test_obj, "test")
