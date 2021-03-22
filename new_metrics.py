import os
import sys
import numpy as np
# from pypinyin import Style, lazy_pinyin
from LAC import LAC
from data import PUNCS

pos_hub_name = ["n", "PER", "nr", "LOC", "ns", "ORG", "nt", "nw", "nz", "TIME", "t",
                "f", "s", "v", "vd", "vn", "a", "ad", "an", "d", "m", "q", "p", "c", "r", "u", "xc", "w"]

lac = LAC(mode="lac")

# 标准类别, 对应回答结果
# 自定义类别形式呢


def eval_tpl(sents1, sents2):
    n = 0.
    # 需要改成myers后的结果, 把对位的先比掉
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]
    for i, x in enumerate(sents1):
        y = sents2[i]
        if len(x) != len(y):
            continue
        px, py = [], []
        for w in x:
            if w in PUNCS:
                px.append(w)
        for w in y:
            if w in PUNCS:
                py.append(w)
        if px == py:
            n += 1
    p = n / len(sents2)
    r = n / len(sents1)
    f = 2 * p * r / (p + r + 1e-16)

    return p, r, f, n, len(sents1), len(sents2)


def get_lac_result(sents):
    return lac.run(sents)


def eval_analogy(sents1, sents2, tpl_sents):
    # 完美情形, 应该是字完全相同,
    # 就算不相同, 也应该是词性相同, 但是我完全想不明白这里为什么不行
    # 准确率（accuracy）： (TP + TN)/(TP + FP + TN + FN)
    # 精准率（precision）：TP / (TP + FP)，正确预测为正占全部预测为正的比例
    # 召回率（recall）： TP / (TP + FN)，正确预测为正占全部正样本的比例
    # 暂时不确定, 这块的准确召回是否正确
    n = 0.
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]
    lac_result1 = get_lac_result(sents1)
    lac_result2 = get_lac_result(sents2)

    # words1, poses1
    # words2, poses2
    n1, n2 = 0., 0.
    for i in range(len(sents1)):
        words1, poses1 = lac_result1[i]
        words2, poses2 = lac_result2[i]
        left = 0
        other1, other2 = 0, 0
        total1, total2 = 0, 0
        for j in range(len(words1)):
            add = len(words1[j])
            # 补充的位置和词性是对的
            if tpl_sents[i][left:left+add] == "_"*add:
                if j < len(words2) and poses1[j] == poses2[j]:
                    n += 1
            else:
                other1 += 1
            left += add
        total1 = len(words1)
        # 我也不知道该咋算了, 直接随便整一个吧
        for j in range(len(words2)):
            add = len(words2[j])
            # 补充的位置和词性是对的
            if tpl_sents[i][left:left+add] == "_"*add:
                if j < len(words1) and poses1[j] == poses2[j]:
                    n += 1
            else:
                other2 += 1
            left += add
        total2 = len(words2)
        # 为空的词数量
        # 对应的槽位都是正确的话, 就没办法起到检测的作用了

        n1 += total1 - other1
        n2 += total2 - other2
    # print(n1, n2, n)
    p = n / (n2 + 1e-16)
    r = n / (n1 + 1e-16)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, f1, n, n1, n2


def eval_endings(sents1, sents2):
    n = 0.
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]

    sents0 = []
    for si, sent1 in enumerate(sents1):
        sent2 = sents2[si]
        if len(sent2) <= len(sent1):
            sents0.append(sent2)
        else:
            sents0.append(sent2[:len(sent1) - 1] + sent1[-1])

    sent = "</s>".join(sents0)
    return sent


def eval(res_file, fid):
    docs = []
    with open(res_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fs = line.split("\t")
            if len(fs) != 3:
                print("error", len(fs), line[:10])
                continue
            x, y, z = fs
            docs.append((x, y, z))

    print(len(docs), res_file, fid)

    ugrams_ = []
    bigrams_ = []
    p_, r_, f1_ = 0., 0., 0.
    n0_, n1_, n2_ = 0., 0., 0.

    p__, r__, f1__ = 0., 0., 0.
    n0__, n1__, n2__ = 0., 0., 0.
    d1_, d2_ = 0., 0.
    d4ends = []

    for x, content, y in docs:
        topic, tpl = x.split("<s2>")
        new_topic, topic = topic.split("<s1>")
        sents1 = content.split("</s>")
        y = y.replace("<bos>", "")
        sents2 = y.split("</s>")
        sents1_ = []
        for sent in sents1:
            sent = sent.strip()
            if sent:
                sents1_.append(sent)
        sents1 = sents1_
        sents2_ = []
        for sent in sents2:
            sent = sent.strip()
            if sent:
                sents2_.append(sent)
        sents2 = sents2_
        tpl_sents = tpl.split("</s>")
        # 这个似乎没啥问题, 但是好像评价的时候应该用模板啊, 直接用content, 也行吧
        p, r, f1, n0, n1, n2 = eval_tpl(sents1, sents2)
        p_ += p
        r_ += r
        f1_ += f1
        n0_ += n0
        n1_ += n1
        n2_ += n2

        # 结构相关贡献值? 就是应该在一个范围吧, 不能太高的
        ugrams = [w for w in ''.join(sents2)]
        bigrams = []
        for bi in range(len(ugrams) - 1):
            bigrams.append(ugrams[bi] + ugrams[bi+1])
        d1_ += len(set(ugrams)) / float(len(ugrams))
        d2_ += len(set(bigrams)) / float(len(bigrams))
        ugrams_ += ugrams
        bigrams_ += bigrams

        p, r, f1, n0, n1, n2 = eval_analogy(sents1, sents2, tpl_sents)
        p__ += p
        r__ += r
        f1__ += f1
        n0__ += n0
        n1__ += n1
        n2__ += n2

        d4end = eval_endings(sents1, sents2)
        d4ends.append(new_topic + "<s1>" + topic + "<s2>" + d4end)

    tpl_macro_p = p_ / len(docs)
    tpl_macro_r = r_ / len(docs)
    tpl_macro_f1 = 2 * tpl_macro_p * tpl_macro_r / (tpl_macro_p + tpl_macro_r)
    tpl_micro_p = n0_ / n2_
    tpl_micro_r = n0_ / n1_
    tpl_micro_f1 = 2 * tpl_micro_p * tpl_micro_r / (tpl_micro_p + tpl_micro_r)

    rhy_macro_p = p__ / len(docs)
    rhy_macro_r = r__ / len(docs)
    rhy_macro_f1 = 2 * rhy_macro_p * rhy_macro_r / (rhy_macro_p + rhy_macro_r)
    rhy_micro_p = n0__ / n2__
    rhy_micro_r = n0__ / n1__
    rhy_micro_f1 = 2 * rhy_micro_p * rhy_micro_r / (rhy_micro_p + rhy_micro_r)

    macro_dist1 = d1_ / len(docs)
    macro_dist2 = d2_ / len(docs)
    micro_dist1 = len(set(ugrams_)) / float(len(ugrams_))
    micro_dist2 = len(set(bigrams_)) / float(len(bigrams_))

    with open("./results_4ending/res4end" + str(fid) + ".txt", "w") as fo:
        for line in d4ends:
            fo.write(line + "\n")
    return tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2


tpl_macro_f1_, tpl_micro_f1_, rhy_macro_f1_, rhy_micro_f1_,  \
    macro_dist1_, micro_dist1_, macro_dist2_, micro_dist2_ = [], [], [], [], [], [], [], []
abalation = "top-32"
for i in range(5):
    f_name = "./results/"+abalation+"/out" + str(i+1)+".txt"
    if not os.path.exists(f_name):
        continue
    tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2 = eval(
        f_name, i + 1)
    print(tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1,
          macro_dist1, micro_dist1, macro_dist2, micro_dist2)
    tpl_macro_f1_.append(tpl_macro_f1)
    tpl_micro_f1_.append(tpl_micro_f1)
    rhy_macro_f1_.append(rhy_macro_f1)
    rhy_micro_f1_.append(rhy_micro_f1)
    macro_dist1_.append(macro_dist1)
    micro_dist1_.append(micro_dist1)
    macro_dist2_.append(macro_dist2)
    micro_dist2_.append(micro_dist2)

print()
print("tpl_macro_f1", np.mean(tpl_macro_f1_), np.std(tpl_macro_f1_, ddof=1))
print("tpl_micro_f1", np.mean(tpl_micro_f1_), np.std(tpl_micro_f1_, ddof=1))
print("rhy_macro_f1", np.mean(rhy_macro_f1_), np.std(rhy_macro_f1_, ddof=1))
print("rhy_micro_f1", np.mean(rhy_micro_f1_), np.std(rhy_micro_f1_, ddof=1))
print("macro_dist1", np.mean(macro_dist1_), np.std(macro_dist1_, ddof=1))
print("micro_dist1", np.mean(micro_dist1_), np.std(micro_dist1_, ddof=1))
print("macro_dist2", np.mean(macro_dist2_), np.std(macro_dist2_, ddof=1))
print("micro_dist2", np.mean(micro_dist2_), np.std(micro_dist2_, ddof=1))
