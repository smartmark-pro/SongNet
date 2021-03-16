from hashlib import new
import json
import collections
import sys
import re
from collections import Counter
import random
import pandas as pd
from collections import Counter
cnt = Counter()
random.seed(10)

example_df = pd.read_excel(
    "./data/new_examples.xlsx")
# /content/drive/MyDrive/examples.xlsx
# ./data/new_examples.xlsx

print(example_df.shape)

data = example_df.to_dict("records")

category_data = collections.defaultdict(list)


def remove_nt(s):
    return str(s).replace("\n", "").replace("\t", "")


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

    return remove_nt("".join(new_text)), remove_nt("".join(new_ftext)), c


c = 0

for i, item in enumerate(data):
    # 需要把每一句对上加一个</s>
    # 把标点直接替换成
    new_text, new_format, c2 = add_split_tag(
        str(item["标准化内容"]), str(item["标准化模板"]))
    item["仿写对象"] = remove_nt(item["仿写对象"])
    item["梗和主题"] = remove_nt(item["梗和主题"])

    text = ("{}<s1>{}<s2>{}".format(
        item["仿写对象"], item["梗和主题"], new_format), new_text)
    cnt.update(list(item["仿写对象"]))
    cnt.update(list(item["梗和主题"]))
    cnt.update(list(remove_nt(item["标准化内容"])))

    if len(new_format) != len(new_text):
        print(i, c, c2, len(new_format), len(new_text),
              new_format[:100], new_text[:100])
        c += 1

    category_data[item["梗和主题"]].append(text)

# 类别数量, 有没有
print("类别数量, 错误结果", len(category_data), c)

all, tc, sulian, sulianc, not_select = 0, 0, 0, 0, 0
tokens, sentences = collections.defaultdict(int), collections.defaultdict(int)
new_category_data = collections.defaultdict(list)
for k, v in category_data.items():
    if len(v) >= 4:  # 一个梗至少有四个仿写

        tc += 1
        all += len(v)
        new_category_data[k] = v
        if k.startswith("苏联"):
            sulian += 1
            sulianc += len(v)
        for gen in v:
            tokens[k] += len(gen[1])
            # print(type(gen), len(gen))
            a = re.findall("</s>", gen[1])
            print(len(a))
            sentences[k] += len(a)
        print(k, len(v), end="\t")
    else:
        # print("少于四条", k, len(v))
        not_select += 1

# 梗数量, 条数
print()
print("梗数量, 条数", tc, all, sulian, sulianc, not_select)

# 想办法直接打印这张表, 手工改, 肯定时太麻烦了


path = "/home/mark/github.com/repeat_gen/"
data_path = path + "data/"
out_path = path + "output/"
origin_file = data_path + "repeat_key.json"
df = pd.read_excel(data_path + "苏联笑话集合.xlsx")
find_data = df.to_dict("records")
find_dict = {
    "卢本伟, 赌怪": "是什么概念我们一般只会用两个字来形容这种",
    "b站, 后浪": "应该看着你们；    像我一样，我看着你们，满怀羡慕。",
    "余光中, 乡愁": "小时候，我在这头, 在那头。",
    "美人鱼, 警局报案": "我要说的事, 你们千万别害怕    - 我们是，我们不会怕，您请说    - 我刚才，",
    "乳法笑话, 新闻标题": "在登陆 向逼近 进入 占领 接近 于今日抵达",
    "倪萍, 五千五百万": "同志们, 这是什么, 四舍五入 将近一个呀",
    "鲁迅, 孔乙己": "店内外充满了快活的空气",
    "马保国, 武德": "我说怎么回事，给我发了一几张截图，我一看",
    "苏联笑话, 好笑的事": " 对面办公桌的同事奇怪的问道：“有什么好笑的事吗？”",
    "苏联笑话, 谁的烟头": "不满的  说：“这是谁的 看了看四周，欣喜的说",
    "苏联笑话, 坐牢": "的一间牢房里关了三个人，彼此间谈起坐牢的原因",
    "苏联笑话, 支持": "主持人慌忙说：那请您赶快坐到主席台上来",
    "苏联笑话, 排队买香肠": "过了约会的时间才到。  ——“对不起，我去来着。”什么是排队",
    "让子弹飞, 惊喜": "给我翻译一下什么TMD叫惊喜",
    "流浪星球, 交通安全宣传": "千万条，第一条，不，两行泪",
    "德国牧师马丁·内莫勒, 我没有说话": "起初他们的时候, 我没有说话因为我",
    "南宁采访偷电动车惯犯, 称看守所比家好": "__是不可能__的, 这辈子都不可能__的, 说话又好听, 超喜欢这里",
    "狂人日记鲁迅, 吃人": "我横竖睡不着，仔细看了半夜，才从字缝里看出字来，满本都写着两个字是",
    "邪不胜正, 正经人谁写日记": "正经人谁 谁能把心里话里？的哪能叫心里话",
    "让子弹飞, 我就是想站着还把钱挣了": "我大老远的来一趟，就是为了看的脸色",
}
with open(origin_file) as f:

    origin_dict = json.load(f)

print(len(origin_dict), len(find_dict))  # 有一些当前没有获取

for item in find_data:
    find_dict[item["梗和主题"]] = item["查找关键词"]
    origin_dict[item["梗和主题"]] = item["文本内容"]
sulian_count = 0
for key in origin_dict:
    if key.startswith("苏联笑话"):
        sulian_count += 1
print("梗和主题", len(origin_dict), sulian_count, len(tokens))


def get_paper_table():
    table_print_keys = ["类别名", "条数", "句子数目", "字符数",
                        "原梗句子数", "原梗字符数", "每条平均句子数", "每条平均字符数"]
    table_text = ""
    tag1 = "\t"
    tag2 = ""  # "&"
    tag3 = "\n"  # "\\\\ \\hline\n"  #
    left, right = [], []
    for k, v in new_category_data.items():
        if k.startswith("苏联笑话"):

            right.append((k, len(v)))
        else:
            left.append((k, len(v)))
    left.sort(key=lambda x: (-x[1], -len(x[0])))
    right.sort(key=lambda x: (-x[1], -len(x[0])))

    table_text += table_print_keys[0]
    left = left[:5]
    right = right[:0]
    for k, count in left:
        table_text += tag1 + tag2 + str(k)

    for k, count in right:
        table_text += tag1 + tag2 + str(k)

    table_text += tag3

    table_text += table_print_keys[1]
    for k, count in left:
        table_text += tag1 + tag2 + str(count)

    for k, count in right:
        table_text += tag1 + tag2 + str(count)

    table_text += tag3

    table_text += table_print_keys[2]
    for k, count in left:
        table_text += tag1 + tag2 + str(sentences[k])

    for k, count in right:
        table_text += tag1 + tag2 + str(sentences[k])

    table_text += tag3

    table_text += table_print_keys[3]
    for k, count in left:
        table_text += tag1 + tag2 + str(tokens[k]-4*sentences[k])

    for k, count in right:
        table_text += tag1 + tag2 + str(tokens[k]-4*sentences[k])

    table_text += tag3

    def get_text_sentence(text):
        ans = 0
        for t in text:
            if t in ["？", "！", "，", "。"]:
                ans += 1
        if ans == 0:
            text = text.replace("\n\n", "\n")
            text = text.replace("\n\n", "\n")
            text = text.replace("\n\n", "\n")
            text = text.replace("\n\n", "\n")
            text = text.replace("\n\n", "\n")
            text = text.replace("\n\n", "\n")
            text = text.replace("\n", "。")
            for t in text:
                if t in ["？", "！", "，", "。"]:
                    ans += 1

        return ans

    table_text += table_print_keys[4]
    for k, count in left:
        table_text += tag1 + tag2 + str(get_text_sentence(origin_dict[k]))

    for k, count in right:
        table_text += tag1 + tag2 + str(get_text_sentence(origin_dict[k]))

    table_text += tag3

    table_text += table_print_keys[5]
    for k, count in left:
        table_text += tag1 + tag2 + str(len(origin_dict[k]))

    for k, count in right:
        table_text += tag1 + tag2 + str(len(origin_dict[k]))

    table_text += tag3

    table_text += table_print_keys[6]
    for k, count in left:
        table_text += tag1 + tag2 + "{:.2f}".format(sentences[k]/count)

    for k, count in right:
        table_text += tag1 + tag2 + "{:.2f}".format(sentences[k]/count)

    table_text += tag3

    table_text += table_print_keys[7]
    for k, count in left:
        table_text += tag1 + tag2 + \
            "{:.2f}".format((tokens[k]-4*sentences[k])/count)

    for k, count in right:
        table_text += tag1 + tag2 + \
            "{:.2f}".format((tokens[k]-4*sentences[k])/count)

    table_text += tag3

    print(table_text)


table_text = get_paper_table()


objects = list(new_category_data.keys())
random.shuffle(objects)
# 8:1:1
train_obj = objects[:int(0.85*len(objects))]
dev_obj = objects[int(0.85*len(objects)):int(0.9*len(objects))]
test_obj = objects[int(0.9*len(objects)):]


def save_data(name, data, tag="r_"):
    with open("./data/{}{}.txt".format(tag, name), "w") as fw:
        for a, b in data:
            fw.write("{}\t{}\n".format(a, b))


def get_data_by_category(keys, name):
    data = []
    for k in keys:
        data.extend(new_category_data[k])
    # 类别, 数量, 取得是先混合后的
    # 可以考虑按类别划分
    print(len(keys), len(data))
    save_data(name, data, "2_")
    return data


train_data = get_data_by_category(train_obj, "train")
dev_data = get_data_by_category(dev_obj, "dev")
test_data = get_data_by_category(test_obj, "test")

# 一会再确定下, dev到底有没有用


def get_data_by_mount(data, name):
    # 类别, 数量, 取得是先混合后的
    # 每个类别都有
    print(len(data))
    save_data(name, data, "1_")
    return data


new_train_data = []
new_dev_data = []
new_test_data = []
for k, v in new_category_data.items():
    random.shuffle(v)
    if len(v) < 10:

        new_train_data += v[:-2]
        new_dev_data.append(v[-2])
        new_test_data.append(v[-1])
    else:
        new_train_data += v[:int(0.88*len(v))]
        new_dev_data += v[int(0.88*len(v)):int(0.92*len(v))]
        new_test_data += v[int(0.92*len(v)):]

    # print(len(new_train_data), len(new_dev_data), len(new_test_data))
new_train_data = get_data_by_mount(new_train_data, "train")
new_dev_data = get_data_by_mount(new_dev_data, "dev")
new_test_data = get_data_by_mount(new_test_data, "test")

print("vocab")
print(len(cnt))  # 5310 + 1024+9 + 64 = 6407 + 几个字.
with open('./data/vocab.txt', 'w', encoding='utf8') as f:
    for x, y in cnt.most_common():
        f.write(x + '\t' + str(y) + '\n')
print("done")
