
import pandas as pd
import os
from new_metrics import new_eval_tpl, eval_disverse, eval_analogy, get_clear_sent, get_sent2_tpl


def get_result_excel(abalation, result_name):
    """人工分析结果, 做一些简单的整理和标记"""
    data = []
    for i in range(5):
        f_name = "./results/"+abalation+"/out" + str(i+1)+".txt"
        if not os.path.exists(f_name):
            continue

        docs = []
        with open(f_name) as f:
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

        for i, (x, content, y) in enumerate(docs):
            topic, tpl = x.split("<s2>")
            new_topic, topic = topic.split("<s1>")
            # sents1 = content.split("</s>")
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
            tpl_p, tpl_r, tpl_f1, tpl_n0, tpl_n1, tpl_n2 = new_eval_tpl(
                sents1, sents2, sents1_tpl)  # 直接通过模板得到的仍然会偏低

            # 结构相关贡献值? 就是应该在一个范围吧, 不能太高的
            d1, d2, ugrams, bigrams = eval_disverse(diff_s2)

            p, r, f1, n0, n1, n2 = eval_analogy(sents1, sents2, sents1_tpl)
            data.append((i, topic, new_topic, tpl, content, y,
                         tpl_p, tpl_r, tpl_f1, tpl_n0, tpl_n1, tpl_n2,
                         d1, d2, p, r, f1, n0, n1, n2))
    df = pd.DataFrame(data, columns=["序号", "原对象", "新对象", "模板", "标准内容", "预测内容",
                                     "模板评估p", "模板评估r", "模板评估f1", "模板评估n0", "模板评估n1", "模板评估n2",
                                     "多样性评估d1", "多样性评估d2",
                                     "类比评估p", "类比评估r", "类比评估f1", "类比评估n0", "类比评估n1", "类比评估n2",
                                     ])
    df.to_excel("./results/{}.xlsx".format(result_name))


if __name__ == '__main__':
    get_result_excel("other1-top-32", "other1")
    get_result_excel("other2-top-32", "other2")
