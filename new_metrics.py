import os
import sys
import numpy as np
import myers
# from pypinyin import Style, lazy_pinyin
from LAC import LAC
from numpy.lib.function_base import diff
from paddle.fluid.data import data
from typing_extensions import final
from data import PUNCS

pos_hub_name = ["n", "PER", "nr", "LOC", "ns", "ORG", "nt", "nw", "nz", "TIME", "t",
                "f", "s", "v", "vd", "vn", "a", "ad", "an", "d", "m", "q", "p", "c", "r", "u", "xc", "w"]

final_pos = {i: i for i in pos_hub_name}
final_pos["PER"] = "nr"
final_pos["LOC"] = "ns"
final_pos["ORG"] = "nt"
final_pos["TIME"] = "t"
final_pos["vd"] = "d"
final_pos["vn"] = "n"
final_pos["ad"] = "d"
final_pos["an"] = "n"


lac = LAC(mode="lac")

# 标准类别, 对应回答结果
# 自定义类别形式呢


def new_eval_tpl2(sents1, sents2):
    n = 0.
    # 原来统计标点是因为,只有一个韵脚
    # 当前实际应当统计模板字符是否相等
    if len(sents1) > len(sents2):
        # 因为存在误差的情况
        # for i in range(len(sents2)):
        #     if
        sents1 = sents1[:len(sents2)]
    for i in range(len(sents1)):
        if sents1[i] == sents2[i]:
            n += 1

    p = n / len(sents2)
    r = n / len(sents1)
    f = 2 * p * r / (p + r + 1e-16)

    return p, r, f, n, len(sents1), len(sents2)


def new_eval_tpl(sents1, sents2, tpl, tag="_"):
    n = 0.
    # 原来统计标点是因为,只有一个韵脚
    # 当前实际应当统计模板字符是否相等
    s1 = "".join(sents1)
    s2 = "".join(sents2)
    # tpls = "".join(tpl)
    # if len(s1) != len(s2):
    #     print(len(s1), len(s2), len(tpls))
    i, k, n = 0, 0, 0
    while i < len(tpl):
        t1, t2 = 0, 0
        for t in tpl[i]:
            if t != tag:
                if k < len(s1):
                    if t == s1[k]:
                        t1 += 1
                if k < len(s2):
                    if t == s2[k]:
                        t2 += 1
            k += 1
        i += 1
        if t1 == t2:
            n += 1

    if len(sents1) > len(sents2):
        # 因为存在误差的情况
        sents1 = sents1[:len(sents2)]

    p = n / len(sents2)
    r = n / len(sents1)
    f = 2 * p * r / (p + r + 1e-16)

    return p, r, f, n, len(sents1), len(sents2)


def eval_tpl(sents1, sents2, tpl):
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


def get_final_pos(pos):
    res = []
    for i in pos:
        res.append(final_pos[i])
    return res


def get_words_pos_same_length(pos1, pos2):
    pos1 = get_final_pos(pos1)
    pos2 = get_final_pos(pos2)
    diff = myers.diff(pos1, pos2)
    # print(len(diff), len(text1), len(text2))
    # print("".join(text1), text1, pos1)
    # print("".join(text2), text2, pos2)
    i = 0
    for (op, t) in diff:
        if op == "k":
            i += 1
        elif op == "":
            pass
    # 近似结果
    return i


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
                if j < len(words2) and final_pos[poses1[j]] == final_pos[poses2[j]]:
                    n += add
            else:
                other1 += add
            left += add
        total1 = len(sents1[i])
        # 我也不知道该咋算了, 直接随便整一个吧
        for j in range(len(words2)):
            add = len(words2[j])
            # 补充的位置和词性是对的
            if tpl_sents[i][left:left+add] == "_"*add:
                # if j < len(words1) and poses1[j] == poses2[j]:
                #     n += 1
                pass
            else:
                other2 += 1
            left += add
        total2 = len(sents2[i])
        # 为空的词数量
        # 对应的槽位都是正确的话, 就没办法起到检测的作用了

        n1 += total1 - other1
        n2 += total2 - other2
    # print(n1, n2, n)
    p = n / (n2 + 1e-16)
    r = n / (n1 + 1e-16)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, f1, n, n1, n2


# def eval_analogy3(sents1, sents2, tpl_sents, tag="_"):
#     # 感觉怎么搞都不太行, 预测的水平过低
#     p = n / (n2 + 1e-16)
#     r = n / (n1 + 1e-16)
#     f1 = 2 * p * r / (p + r + 1e-16)
#     return p, r, f1, n, n1, n2

def eval_analogy2(sents1, sents2, tpl_sents, tag="_"):
    """只能得到一个较为模糊的值, 最长公共字串似乎会好一点, 统计字数, 这里是词的数量, 还是不太行啊"""
    n = 0.
    # 应该
    s1 = "".join(sents1)
    s2 = "".join(sents2)
    _, poses1 = get_lac_result(s1)
    _, poses2 = get_lac_result(s2)

    n = get_words_pos_same_length(poses1, poses2)
    common_words = []
    tpls = "".join(tpl_sents)
    for i in tpls:
        if i != tag:
            common_words.append(i)
    _, tpl_poses = get_lac_result("".join(common_words))
    n -= len(tpl_poses)
    n2 = len(poses2)
    n1 = len(poses1)
    p = n / (n2 + 1e-16)
    r = n / (n1 + 1e-16)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, f1, n, n1, n2


# def eval_endings(sents1, sents2):
#     n = 0.
#     if len(sents1) > len(sents2):
#         sents1 = sents1[:len(sents2)]

#     sents0 = []
#     for si, sent1 in enumerate(sents1):
#         sent2 = sents2[si]
#         if len(sent2) <= len(sent1):
#             sents0.append(sent2)
#         else:
#             sents0.append(sent2[:len(sent1) - 1] + sent1[-1])

#     sent = "</s>".join(sents0)
#     return sent


def align_part(origin, content, mask_tag="_"):
    # 会得到一个不错的模板

    diff = myers.diff(origin, content)
    rs = []
    count = 0

    for op, t in diff:
        if op == "k":
            # 保留
            # 这个相当于是共同的东西,
            rs.append(t)  # "_"
            count += 1
        elif op == "o":
            # 忽略掉
            continue
        elif op == "i":
            # 插入 就是新的内容
            rs.append(mask_tag)  # t
            # count += 1

        else:
            # 删除
            # rs += mask_tag
            # 也不要了
            continue
    return rs, count


def get_sent2_tpl(sents2, tpl_sents):
    s2 = "".join(sents2)
    tpl = "".join(tpl_sents)
    tpl = "".join(tpl.split("_"))

    new_tpl, c = align_part(tpl, s2)  # 会得到一个和s2完全一样长的模板
    assert len(new_tpl) == len(s2)
    common_s2, diff_s2, sents2_tpl = [], [], []
    k = 0
    for i in range(len(sents2)):
        c, d, t = [], [], []
        for j in range(len(sents2[i])):
            if new_tpl[k] == sents2[i][j]:
                c.append(sents2[i][j])
            else:
                d.append(sents2[i][j])
            t.append(new_tpl[k])
            k += 1
        common_s2.append("".join(c))
        diff_s2.append("".join(d))
        sents2_tpl.append("".join(t))
    assert len(common_s2) == len(diff_s2)
    return common_s2, diff_s2, new_tpl, sents2_tpl


def eval_disverse(sents2):

    ugrams = [w for w in ''.join(sents2)]

    bigrams = []
    for bi in range(len(ugrams) - 1):
        bigrams.append(ugrams[bi] + ugrams[bi+1])
    if len(ugrams) == 0 or len(bigrams) == 0:
        return 0, 0, [], []
    d1 = len(set(ugrams)) / float(len(ugrams))
    d2 = len(set(bigrams)) / float(len(bigrams))
    return d1, d2, ugrams, bigrams


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
            # if len(docs) > 5:
            #     break

    # print(len(docs), res_file, fid)

    ugrams_ = []
    bigrams_ = []
    p_, r_, f1_ = 0., 0., 0.
    n0_, n1_, n2_ = 0., 0., 0.

    p__, r__, f1__ = 0., 0., 0.
    n0__, n1__, n2__ = 0., 0., 0.
    d1_, d2_ = 0., 0.
    # d4ends = []

    def get_clear_sent(sents):
        sents_ = []
        for sent in sents:
            sent = sent.strip()
            if sent:
                sents_.append(sent)
        return sents_
    for x, content, y in docs:
        topic, tpl = x.split("<s2>")
        new_topic, topic = topic.split("<s1>")
        sents1 = content.split("</s>")
        y = y.replace("<bos>", "")
        sents2 = y.split("</s>")
        sents1 = get_clear_sent(sents1)
        sents2 = get_clear_sent(sents2)
        sents1_tpl = tpl.split("</s>")
        sents1_tpl = get_clear_sent(sents1_tpl)
        # 需要一个函数将sents2 拆成两个
        common_s2, diff_s2, new_tpl, sents2_tpl = get_sent2_tpl(
            sents2, sents1_tpl)
        # 这个似乎没啥问题, 但是好像评价的时候应该用模板啊, 直接用content, 也行吧
        p, r, f1, n0, n1, n2 = new_eval_tpl(
            sents1, sents2, sents1_tpl)  # 直接通过模板得到的仍然会偏低
        p_ += p
        r_ += r
        f1_ += f1
        n0_ += n0
        n1_ += n1
        n2_ += n2

        # 结构相关贡献值? 就是应该在一个范围吧, 不能太高的
        d1, d2, ugrams, bigrams = eval_disverse(diff_s2)
        d1_ += d1
        d2_ += d2
        ugrams_ += ugrams
        bigrams_ += bigrams

        p, r, f1, n0, n1, n2 = eval_analogy(sents1, sents2, sents1_tpl)
        p__ += p
        r__ += r
        f1__ += f1
        n0__ += n0
        n1__ += n1
        n2__ += n2

        # d4end = eval_endings(sents1, sents2)
        # d4ends.append(new_topic + "<s1>" + topic + "<s2>" + d4end)

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

    # with open("./results_4ending/res4end" + str(fid) + ".txt", "w") as fo:
    #     for line in d4ends:
    #         fo.write(line + "\n")
    return tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2


def get_metrics_tabel(abalation):
    print(abalation)
    tpl_macro_f1_, tpl_micro_f1_, rhy_macro_f1_, rhy_micro_f1_,  \
        macro_dist1_, micro_dist1_, macro_dist2_, micro_dist2_ = [], [], [], [], [], [], [], []
    # abalation = "top-32"
    for i in range(1):
        f_name = "./results/"+abalation+"/out" + str(i+1)+".txt"
        if not os.path.exists(f_name):
            continue
        tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2 = eval(
            f_name, i + 1)
        # print(tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1,
        #       macro_dist1, micro_dist1, macro_dist2, micro_dist2)
        tpl_macro_f1_.append(tpl_macro_f1)
        tpl_micro_f1_.append(tpl_micro_f1)
        rhy_macro_f1_.append(rhy_macro_f1)
        rhy_micro_f1_.append(rhy_micro_f1)
        macro_dist1_.append(macro_dist1)
        micro_dist1_.append(micro_dist1)
        macro_dist2_.append(macro_dist2)
        micro_dist2_.append(micro_dist2)

    """\multirow{2}*{test1} & model1 & 2.377 & 236 & 171 & 2.377 & 236 & 171 & 236 & 171\\
        \cline {2-10} & model2 & 2.377 & 236 & 171 & 2.377 & 236 & 171 & 236 & 171\\
    \multirow{2}*{test2} & model1 & 2.377 & 236 & 171 & 2.377 & 236 & 171 & 236 & 171\\
    \cline {2-10} & model2 & 2.377 & 236 & 171 & 2.377 & 236 & 171 & 236 & 171\\"""
    data = [macro_dist1_, micro_dist1_,
            macro_dist2_, micro_dist2_, tpl_macro_f1_, tpl_micro_f1_, rhy_macro_f1_, rhy_micro_f1_]
    for scores in data:
        print("& {:.2f} ".format(np.mean(scores)*100), end="")
    print("\\\\")
    print()

    # print("tpl_macro_f1", np.mean(tpl_macro_f1_), np.std(tpl_macro_f1_, ddof=1))
    # print("tpl_micro_f1", np.mean(tpl_micro_f1_), np.std(tpl_micro_f1_, ddof=1))
    # print("rhy_macro_f1", np.mean(rhy_macro_f1_), np.std(rhy_macro_f1_, ddof=1))
    # print("rhy_micro_f1", np.mean(rhy_micro_f1_), np.std(rhy_micro_f1_, ddof=1))
    # print("macro_dist1", np.mean(macro_dist1_), np.std(macro_dist1_, ddof=1))
    # print("micro_dist1", np.mean(micro_dist1_), np.std(micro_dist1_, ddof=1))
    # print("macro_dist2", np.mean(macro_dist2_), np.std(macro_dist2_, ddof=1))
    # print("micro_dist2", np.mean(micro_dist2_), np.std(micro_dist2_, ddof=1))
# get_metrics_tabel("top-32-test1-1-2999")
# get_metrics_tabel("top-32-test1-2-2999")


get_metrics_tabel("other1-top-32")
get_metrics_tabel("other2-top-32")
# other1-top-32
# get_metrics_tabel("top-32-test1-1-2999")
# get_metrics_tabel("other2-top-32")
