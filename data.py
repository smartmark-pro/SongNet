import random
import torch
import numpy as np
from LAC import LAC

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BOC, EOC = '<boc>', '<eoc>'
LS, RS, SP = '<s>', '</s>', ' '
CS = ['<c-1>'] + ['<c' + str(i) + '>' for i in range(32)]  # content
SS = ['<s-1>'] + ['<s-2>'] + \
    ['<s' + str(i) + '>' for i in range(512)]  # segnment
PS = ['<p-1>'] + ['<p-2>'] + \
    ['<p' + str(i) + '>' for i in range(512)]  # position
TS = ['<t-1>'] + ['<t-2>'] + ['<t' + str(i) + '>' for i in range(32)]

# 将词性引入到模板中, 相同的词跳过作为原样
#
pos_hub_name = ["n", "PER", "nr", "LOC", "ns", "ORG", "nt", "nw", "nz", "TIME", "t",
                "f", "s", "v", "vd", "vn", "a", "ad", "an", "d", "m", "q", "p", "c", "r", "u", "xc", "w"]
pos2CS_dict = {pos_hub_name[i]: CS[i+4] for i in range(len(pos_hub_name))}
print(len(pos2CS_dict), pos2CS_dict)

PUNCS = set([",", ".", "?", "!", ":", "，", "。", "？", "！", "："])

BUFSIZE = 4096000
lac = LAC(mode='lac')


def ListsToTensor(xs, vocab=None):
    # 通过这个函数, 变成索引, 顺便对空的padding
    # max
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len - len(x))
        else:
            y = x + [0]*(max_len - len(x))
        ys.append(y)
    return ys


def _back_to_text_for_check(x, vocab):
    w = x.t().tolist()
    for sent in vocab.idx2token(w):
        print(' '.join(sent))


def batchify(data, vocab):
    xs_tpl, xs_seg, xs_pos, \
        ys_truth, ys_inp, \
        ys_tpl, ys_seg, ys_pos, msk = [], [], [], [], [], [], [], [], []
    for xs_tpl_i, xs_seg_i, xs_pos_i, ys_i, ys_tpl_i, ys_seg_i, ys_pos_i in data:
        xs_tpl.append(xs_tpl_i)
        xs_seg.append(xs_seg_i)
        xs_pos.append(xs_pos_i)

        ys_truth.append(ys_i)
        ys_inp.append([BOS] + ys_i[:-1])
        ys_tpl.append(ys_tpl_i)
        ys_seg.append(ys_seg_i)
        ys_pos.append(ys_pos_i)

        msk.append([1 for i in range(len(ys_i))])

    xs_tpl = torch.LongTensor(ListsToTensor(xs_tpl, vocab)).t_().contiguous()
    xs_seg = torch.LongTensor(ListsToTensor(xs_seg, vocab)).t_().contiguous()
    xs_pos = torch.LongTensor(ListsToTensor(xs_pos, vocab)).t_().contiguous()
    ys_truth = torch.LongTensor(ListsToTensor(
        ys_truth, vocab)).t_().contiguous()
    ys_inp = torch.LongTensor(ListsToTensor(ys_inp, vocab)).t_().contiguous()
    ys_tpl = torch.LongTensor(ListsToTensor(ys_tpl, vocab)).t_().contiguous()
    ys_seg = torch.LongTensor(ListsToTensor(ys_seg, vocab)).t_().contiguous()
    ys_pos = torch.LongTensor(ListsToTensor(ys_pos, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(
        msk)).t_().contiguous()  # 一维数组, 存储连续性, msk纯1?
    # for i in (xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk):
    #     print(i.shape, end="\t")
    # print()
    return xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk


def s2t(strs, vocab):
    inp, msk = [], []
    for x in strs:
        inp.append(x)
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


def s2xy(lines, vocab, max_len, min_len):
    data = []
    c = 0
    for line in lines:
        res = parse_line(line, max_len, min_len)
        if not res:
            continue
        data.append(res)
    return batchify(data, vocab)


def new_s2xy(lines, vocab, max_len, min_len):
    data = []
    c1, c2 = 0, 0
    for line in lines:
        res = gen_parse_line(line, max_len, min_len)
        if not res:
            continue
        for p in res:
            data.append(p)
            c2 += 1
        c1 += 1

    # print(len(data), c1, c2)  # 记一下数字
    return batchify(data, vocab)


def new_word_s2xy(lines, vocab, max_len, min_len):
    data = []
    c1, c2 = 0, 0
    for line in lines:
        res = word_gen_parse_line(line, max_len, min_len)
        if not res:
            continue
        for p in res:
            data.append(p)
            c2 += 1
        c1 += 1

    # print(len(data), c1, c2)  # 记一下数字
    return batchify(data, vocab)


def word_gen_parse_line(line, max_len, min_len, bound=300):
    # 每个字符都返回list类型,
    # 是不是应该模板改成带有词性的模板
    # 不需要变得直接原文, <pos-xx>
    # 明白了, 直接按照<\s> 分割, 然后再重新拼起来, 这样, 就能得到正确的结果了
    # 把tpl也这样拼起来, 尽管这样有点搞笑
    line, text = line.strip().split("\t")
    # print(len(line), len(text), line, text)
    if not line:
        return []
    fs = line.split("<s2>")
    question, gen_name = fs[0].split("<s1>")
    question_words = lac.run(question)[0]
    gen_name_words = lac.run(gen_name)[0]
    actual_max = max_len - (len(question_words)+len(gen_name_words)+3)

    tpl = fs[1].strip()
    # print(len(text), len(tpl))
    # 模板需要重新合并, 妈呀, 之前是被切开的, 因为字的原因被重组的
    text_list = text.split(RS)
    text_results = lac.run(text_list)  # 这里有"</s>"分词之后怎么搞啊
    words, poses = [], []
    for word, pos in text_results:

        words += word + [RS]
        poses += pos + [RS]
    count = len("".join(words))
    # print(count, len(tpl))
    # print(len(words), len(poses), len(text), len(text_list))
    # print("".join(words))
    # assert count == len(tpl)
    # 为啥不等, 按理说应该完全一样啊

    new_tpl = []
    count = 0
    # 纯文字就用w替代
    for i, w in enumerate(words):
        cur = tpl[count: count + len(w)]
        if cur == w:
            new_tpl.append(w)
        else:
            new_tpl.append(poses[i])
        count += len(w)

    assert len(new_tpl) == len(words)
    # print(words, poses)
    # assert len(tpl) == len(text)
    # print(len(tpl), len(text))
    #超过这部分的长度, 做分段
    # 代码逻辑真的离谱, 不如直接分了
    if len(words) < min_len:
        return []
    if len(words) <= actual_max:
        tpl_array = [(0, len(words))]
    else:
        # 分段必须要按句分, 不能有断开
        # . 的数据处理错误
        n = len(words)
        last_punc = 0
        cur = 0
        tpl_array = []
        for i in range(n):
            if words[i] == RS:
                last_punc = i
            if i-cur != 0 and (i-cur) % actual_max == 0:
                # 可能会出错误, 最后会加一个eos
                tpl_array.append((cur, last_punc))
                cur = last_punc+1
        if cur < n:
            tpl_array.append((cur, n))
    rs = []
    # print([b-a+1 for a, b in tpl_array])

    def get_sents(arr):
        ans = []
        cur = []
        for i in range(len(arr)):
            if arr[i] != RS:
                cur.append(arr[i])
            else:
                ans.append(cur)
                cur = []
        return ans

    for i, (start, end) in enumerate(tpl_array):
        tpl_sents = get_sents(new_tpl[start:end+1])
        sents = get_sents(words[start:end+1])
        # print(tpl_sents, sents) # 分句
        ys = []
        xs_tpl = []
        xs_seg = []
        xs_pos = []

        # 需要增加问题, 类别, 拆分序号.
        # 问题可能增加了一些信息但是也可能带来一些无意信息, 而且是变长的, 这里没法合理的放置
        # idx = "<p-{}>".format(i)
        ws = [w for w in gen_name_words]
        repeater = [q for q in question_words]

        xs_tpl = ws + [i] + [EOC] + repeater + [EOC]
        xs_seg = [SS[0] for _ in ws] + [i] + [EOC] + [SS[1]
                                                      for _ in repeater] + [EOC]
        # 从ss300开始可能是因为倒着的缘故
        # 但是这里用一个值表示, 会不会不能表示位置啊?
        xs_pos = [PS[bound+j] for j in range(
            len(ws))] + [i] + [EOC] + [PS[bound+j] for j in range(len(repeater))] + [EOC]

        ys_tpl = []
        ys_seg = []
        ys_pos = []
        # 按句编码
        for si, sent in enumerate(sents):
            ws = sent
            for k, w in enumerate(sent):
                # ws.append(w)
                if tpl_sents[si][k] in pos2CS_dict:
                    # print(si, k, sent)
                    ys_tpl.append(pos2CS_dict[tpl_sents[si][k]])

                else:
                    # 标点符号, 不可变
                    ys_tpl.append(w)  # 改成一样的polish
            ys += sent + [RS]
            ys_tpl += [RS]
            ys_seg += [SS[si + 2] for w in ws] + [RS]
            ys_pos += [PS[i + 2] for i in range(len(ws))] + [RS]
            # 因为总是韵脚, 所以倒着来更好一些,但是梗仿写不需要
        ys += [EOS]
        ys_tpl += [EOS]
        ys_seg += [EOS]
        ys_pos += [EOS]

        xs_tpl += ys_tpl
        xs_seg += ys_seg
        xs_pos += ys_pos

        # print(ys)
        if len(ys) < min_len:
            continue

        rs.append((xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos))
    return rs


def gen_parse_line(line, max_len, min_len, bound=300):
    line, text = line.strip().split("\t")
    # print(len(line), len(text), line, text)
    if not line:
        return []
    fs = line.split("<s2>")
    question, gen_name = fs[0].split("<s1>")
    tpl = fs[1].strip()
    # assert len(tpl) == len(text)
    # print(len(tpl), len(text))
    #超过这部分的长度, 做分段
    if len(tpl) < min_len:
        return []
    if len(tpl) <= max_len:
        tpl_array = [(0, len(tpl))]
    else:
        # 分段必须要按句分, 不能有断开
        # . 的数据处理错误
        n = len(text)
        last_punc = -1
        cur = 0
        tpl_array = []
        for i in range(n):
            if text[i] in PUNCS:
                last_punc = i+4
            if i > 0 and i % max_len == 0:
                # 可能会出错误, 最后会加一个eos
                tpl_array.append((cur, last_punc))
                cur = last_punc+1
        if cur < len(tpl):
            tpl_array.append((cur, len(tpl)))
    rs = []
    # print(tpl_array)
    for i, (start, end) in enumerate(tpl_array):
        tpl_sents = tpl[start:end+1].split("</s>")
        sents = text[start:end+1].split("</s>")
        # print(tpl_sents, sents) # 分句
        ys = []
        xs_tpl = []
        xs_seg = []
        xs_pos = []

        # 需要增加问题, 类别, 拆分序号.
        # 问题可能增加了一些信息但是也可能带来一些无意信息, 而且是变长的, 这里没法合理的放置
        #
        # idx = "<p-{}>".format(i)
        ws = [w for w in gen_name]
        repeater = [q for q in question]

        # 模板中是数字, 其它是符号index 是不是不对啊
        # 这里再seg, pos 只用seg编码, 应该不算错误吧
        #
        xs_tpl = ws + [i] + [EOC] + repeater + [EOC]
        xs_seg = [SS[0] for _ in ws] + [i] + [EOC] + [SS[1]
                                                      for _ in repeater] + [EOC]
        # 从ss300开始可能是因为倒着的缘故
        # 但是这里用一个值表示, 会不会不能表示位置啊?
        xs_pos = [PS[bound+j] for j in range(
            len(ws))] + [i] + [EOC] + [PS[bound+j] for j in range(len(repeater))] + [EOC]

        ys_tpl = []
        ys_seg = []
        ys_pos = []
        for si, sent in enumerate(sents):
            ws = []
            sent = sent.strip()
            if not sent:
                continue
            # print(sent)
            for k, w in enumerate(sent):
                ws.append(w)
                if w.strip() and w not in PUNCS:
                    # print(si, k, sent)

                    if tpl_sents[si][k] == "_":
                        # 可变化的文字
                        ys_tpl.append(CS[2])
                    else:
                        # 不可变的文字
                        ys_tpl.append(w)
                else:
                    # 标点符号
                    ys_tpl.append(w)  # 改成一样的polish
            ys += ws + [RS]
            # 直接替换
            # if ws[-1] in PUNCS:
            #     ys_tpl[-2] = CS[3]
            # else:
            #     ys_tpl[-1] = CS[3]
            ys_tpl += [RS]
            ys_seg += [SS[si + 2] for w in ws] + [RS]
            ys_pos += [PS[i + 2] for i in range(len(ws))] + [RS]
            # 因为总是韵脚, 所以倒着来更好一些,但是梗仿写不需要
        ys += [EOS]
        ys_tpl += [EOS]
        ys_seg += [EOS]
        ys_pos += [EOS]

        xs_tpl += ys_tpl
        xs_seg += ys_seg
        xs_pos += ys_pos

        # print(ys)
        if len(ys) < min_len:
            return []

        rs.append((xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos))
    return rs


def parse_line(line, max_len, min_len):
    line = line.strip()
    if not line:
        return None
    fs = line.split("<s2>")
    author, cipai = fs[0].split("<s1>")
    sents = fs[1].strip()
    if len(sents) > max_len:
        sents = sents[:max_len]
    if len(sents) < min_len:
        return None
    sents = sents.split("</s>")

    ys = []
    xs_tpl = []
    xs_seg = []
    xs_pos = []

    ctx = cipai
    ws = [w for w in ctx]
    xs_tpl = ws + [EOC]
    xs_seg = [SS[0] for w in ws] + [EOC]
    xs_pos = [SS[i+300] for i in range(len(ws))] + [EOC]

    ys_tpl = []
    ys_seg = []
    ys_pos = []
    for si, sent in enumerate(sents):
        ws = []
        sent = sent.strip()
        if not sent:
            continue
        for w in sent:
            ws.append(w)
            if w.strip() and w not in PUNCS:
                ys_tpl.append(CS[2])
            else:
                ys_tpl.append(CS[1])
        ys += ws + [RS]
        if ws[-1] in PUNCS:  # 不变的是韵脚,
            ys_tpl[-2] = CS[3]
        else:
            ys_tpl[-1] = CS[3]
        ys_tpl += [RS]
        ys_seg += [SS[si + 1] for w in ws] + [RS]
        ys_pos += [PS[len(ws) - i] for i in range(len(ws))] + [RS]

    ys += [EOS]
    ys_tpl += [EOS]
    ys_seg += [EOS]
    ys_pos += [EOS]

    xs_tpl += ys_tpl
    xs_seg += ys_seg
    xs_pos += ys_pos

    if len(ys) < min_len:
        return None
    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos


def s2xy_polish(lines, vocab, max_len, min_len):
    data = []
    for line in lines:
        res = parse_line_polish(line, max_len, min_len)
        data.append(res)
    return batchify(data, vocab)


def parse_line_polish(line, max_len, min_len):
    line = line.strip()
    if not line:
        return None
    fs = line.split("<s2>")
    author, cipai = fs[0].split("<s1>")
    sents = fs[1].strip()
    if len(sents) > max_len:
        sents = sents[:max_len]
    if len(sents) < min_len:
        return None
    sents = sents.split("</s>")

    ys = []
    xs_tpl = []
    xs_seg = []
    xs_pos = []

    ctx = cipai
    ws = [w for w in ctx]
    xs_tpl = ws + [EOC]
    xs_seg = [SS[0] for w in ws] + [EOC]
    xs_pos = [SS[i+300] for i in range(len(ws))] + [EOC]

    ys_tpl = []
    ys_seg = []
    ys_pos = []
    for si, sent in enumerate(sents):
        ws = []
        sent = sent.strip()
        if not sent:
            continue
        for w in sent:
            ws.append(w)
            if w == "_":
                ys_tpl.append(CS[2])
            else:
                ys_tpl.append(w)
        ys += ws + [RS]
        ys_tpl += [RS]
        ys_seg += [SS[si + 1] for w in ws] + [RS]
        ys_pos += [PS[len(ws) - i] for i in range(len(ws))] + [RS]

    ys += [EOS]
    ys_tpl += [EOS]
    ys_seg += [EOS]
    ys_pos += [EOS]

    xs_tpl += ys_tpl
    xs_seg += ys_seg
    xs_pos += ys_pos

    if len(ys) < min_len:
        return None

    return xs_tpl, xs_seg, xs_pos, ys, ys_tpl, ys_seg, ys_pos


class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len_y, min_len_y):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len_y = max_len_y
        self.min_len_y = min_len_y
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):

        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = []
        for line in lines[:-1]:  # the last sent may be imcomplete
            # res = parse_line(line, self.max_len_y, self.min_len_y)
            # if not res:
            #     continue
            # data.append(res)
            res = word_gen_parse_line(line, self.max_len_y, self.min_len_y)
            if not res:
                continue
            for p in res:
                data.append(p)

        random.shuffle(data)

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None):
        idx2token = [PAD, UNK, BOS, EOS] + [BOC, EOC, LS, RS, SP] + CS + SS + PS + TS \
            + (specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try:
                token, cnt = line.strip().split()
            except:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)


if __name__ == "__main__":
    name = ["xs_tpl", "xs_seg", "xs_pos", "ys", "ys_tpl", "ys_seg", "ys_pos"]

    lines = ["吴文英<s1>诉衷情<s2>西风吹鹤到人间。</s>凉月满缑山。</s>银河万里秋浪，重载客槎还。</s>河汉女，巧云鬟。</s>夜阑干。</s>钗头新约，针眼娇颦，楼上秋寒。",
             "蔡楠<s1>诉衷情<s2>夕阳低户水当楼。</s>风烟惨淡秋。</s>乱云飞尽碧山留。</s>寒沙卧海鸥。</s>浑似画，只供愁。</s>相看空泪流。</s>故人如欲问安不。</s>病来今白头。"]

    for line in lines:
        print("line", line)
        rs = parse_line(line, 300, 2)  # _polish
        for i in range(0, len(rs), 3):
            print(name[i], len(rs[i]), rs[i])
        print()

    # line 蔡楠<s1>诉衷情<s2>夕阳低户水当楼。</s>风烟惨淡秋。</s>乱云飞尽碧山留。</s>寒沙卧海鸥。</s>浑似画，只供愁。</s>相看空泪流。</s>故人如欲问安不。</s>病来今白头。
    # xs_tpl['诉', '衷', '情', '<eoc>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>',
    #        '</s>', '<c1>', '<c1>', '<c1>', '<c0>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<eos>']
    # xs_seg['<s-1>', '<s-1>', '<s-1>', '<eoc>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '</s>', '<s1>', '<s1>', '<s1>', '<s1>', '<s1>', '<s1>', '</s>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '</s>', '<s3>', '<s3>', '<s3>', '<s3>', '<s3>',
    #        '<s3>', '</s>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '</s>', '<s5>', '<s5>', '<s5>', '<s5>', '<s5>', '<s5>', '</s>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '</s>', '<s7>', '<s7>', '<s7>', '<s7>', '<s7>', '<s7>', '</s>', '<eos>']
    # xs_pos['<s299>', '<s300>', '<s301>', '<eoc>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>',
    #        '<p0>', '</s>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<eos>']
    # ys['夕', '阳', '低', '户', '水', '当', '楼', '。', '</s>', '风', '烟', '惨', '淡', '秋', '。', '</s>', '乱', '云', '飞', '尽', '碧', '山', '留', '。', '</s>', '寒', '沙', '卧', '海', '鸥', '。', '</s>', '浑',
    #     '似', '画', '，', '只', '供', '愁', '。', '</s>', '相', '看', '空', '泪', '流', '。', '</s>', '故', '人', '如', '欲', '问', '安', '不', '。', '</s>', '病', '来', '今', '白', '头', '。', '</s>', '<eos>']
    # ys_tpl['<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>',
    #        '<c1>', '<c1>', '<c1>', '<c0>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<c1>', '<c1>', '<c1>', '<c1>', '<c2>', '<c0>', '</s>', '<eos>']
    # ys_seg['<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '<s0>', '</s>', '<s1>', '<s1>', '<s1>', '<s1>', '<s1>', '<s1>', '</s>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '<s2>', '</s>', '<s3>', '<s3>', '<s3>', '<s3>', '<s3>', '<s3>', '</s>',
    #        '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '<s4>', '</s>', '<s5>', '<s5>', '<s5>', '<s5>', '<s5>', '<s5>', '</s>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '<s6>', '</s>', '<s7>', '<s7>', '<s7>', '<s7>', '<s7>', '<s7>', '</s>', '<eos>']
    # ys_pos['<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>',
    #        '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p7>', '<p6>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<p5>', '<p4>', '<p3>', '<p2>', '<p1>', '<p0>', '</s>', '<eos>']

    lines = ["""在慈溪中学李晓燕老师班里就读是什么体验？<s1>苏联笑话-6<s2>__举行盛大五一节游行，</s>__率_______全体出席，</s>检阅游行队伍。</s>就在游行队伍通过主席台的时候，</s>_``_同志突然发现人群中有一个人掏出一把___了_几天__的_，</s>于是他马上对身边的______：“我敢打赌，</s>这个_____的人里面没穿内裤！</s>”_____不以为然，</s>难道__同志真长了透视眼不成？</s>_马上命令警卫把那个人叫道跟前，</s>亲自询问，</s>吃惊地发现，</s>这人长裤里面果然是光着的。</s>_______地问领袖：“__同志，</s>您是如何知道透过外衣看见他没穿内裤的？</s>”__回答：“我看见他掏出了新__，</s>他的___显然没用来买内裤嘛。</s>”众人大惊，</s>无不佩服领袖超凡的洞察力……	慈中举行盛大五一节游行，</s>校长率各年级优秀教师全体出席，</s>检阅游行队伍。</s>就在游行队伍通过主席台的时候，</s>校长同志突然发现人群中有一个人掏出一把梳子梳了梳几天没洗的头，</s>于是他马上对身边的miss李道：“我敢打赌，</s>这个拿梳子梳头的人里面没穿内裤！</s>”miss李不以为然，</s>难道校长同志真长了透视眼不成？</s>她马上命令警卫把那个人叫道跟前，</s>亲自询问，</s>吃惊地发现，</s>这人长裤里面果然是光着的。</s>miss李敬佩地问领袖：“校长同志，</s>您是如何知道透过外衣看见他没穿内裤的？</s>”校长回答：“我看见他掏出了新梳子，</s>他的零花钱显然没用来买内裤嘛。</s>”众人大惊，</s>无不佩服领袖超凡的洞察力……""",
             """当英国首相进入ICU的时候<s1>让子弹飞, 惊喜<s2>你给翻译翻译，</s>什么叫____？</s>__</s>___：__________________ 。</s>___：翻译翻译__</s>____难道你听不懂_______？</s> 。</s>___：你_翻译翻译，</s>___________</s>___翻译给我听，</s>什么___叫________</s>！</s>__</s>___：____就是，</s>________，</s>_____________，</s>______的______</s>_____</s>_______________了 。</s>___：翻译翻译！</s>_</s>_</s>___：____就是，</s>_________，</s>___________，</s>____________</s>____</s>______________的___</s>_________</s>______</s>_______！</s>！</s>_</s>________，</s>_______</s>_______</s>_____的	你给翻译翻译，</s>什么叫群体免疫？</s> 。</s>约翰逊：大家一起感染病毒产生抗体获得免疫力啊 。</s>中国人：翻译翻译 。</s>约翰逊：难道你听不懂什么是感染病毒？</s> 。</s>中国人：你给翻译翻译，</s>什么TMD叫群体免疫？</s>我让你翻译给我听，</s>什么TMD叫TMD群体免疫？</s>！</s> 。</s>约翰逊：群体免疫就是，</s>我们不怕感染病毒，</s>感染了在家休息几天就能好了，</s>以后什么变异的新冠都不怕，</s>省钱省事，</s>国家多建设几个太平间火葬场就行了 。</s>中国人：翻译翻译！</s>！</s>！</s>约翰逊：群体免疫就是，</s>我们全国在岛上养蛊，</s>隔着屏幕看你们手忙脚乱，</s>顺便杠一下你们限制自由，</s>不皿煮，</s>等我们大嘤帝国人人都有免疫力的时候，</s>放出去全世界乱串，</s>把你们毒死，</s>重新大航海辉煌！</s>！</s>！</s>要是没把你们毒死，</s>我们只要死人，</s>反正甩锅你们，</s>你们先爆发的""",
             """怼了大学的学生会，将有<em>什么</em>后果？<s1>卢本伟, 赌怪<s2>你们可能不知道只用________是什么概念_</s>我们一般只会用两个字来形容这种人：__！</s>我经常说一句话，</s>当年___他能__________，</s>________________不是问题。</s>埋伏他一手，</s>这个__不能_，</s>这个__不用_，</s>他死定了。</s>反手_____给一个___，</s>__发__。</s>他也__？</s>但是不用怕，</s>他赢不了我_</s>_________，</s>___________，</s>很牛逼这个__，</s>如果把这个__换成____，</s>我这个__将__绝杀，</s>但是换不得。</s>单____，</s>___，</s>直接___。</s>_他_____他。</s>__快点，</s>__，</s>__你__都___吗？</s>__你快点啊！</s>___别磨磨蹭蹭的_</s>_________。</s>___，</s>应该_____的____</s>给__倒杯茶好吧，</s>__给你倒一杯_____</s>给__倒一杯卡布奇诺！</s>开始你的___，</s>_他_他。</s>漂亮！</s>_你______我_？</s>你___我？</s>！</s>你今天______把___了，</s>我！</s>当！</s>场！</s>就把______吃掉	你们可能不知道只用一个人挑战学生会是什么概念。</s>我们一般只会用两个字来形容这种人：头铁！</s>我经常说一句话，</s>当年王思聪他能一个人怼了半个娱乐圈，</s>本大学生一个人怼个学生会（停顿）不是问题。</s>埋伏他一手，</s>这个老师不能怼，</s>这个老师不用管，</s>他死定了。</s>反手等老师走了给一个下马威，</s>大声发闷气。</s>他也怼我？</s>但是不用怕，</s>他赢不了我。</s>列举他们的不当行为，</s>拿出证据让他不能告黑状，</s>很牛逼这个证据，</s>如果把这个部长换成学生会长，</s>我这个证据将完成绝杀，</s>但是换不得。</s>单留一个人，</s>傻～逼，</s>直接怼部长。</s>拿他一个证据怼他。</s>同学快点，</s>同学，</s>同学你部长都不敢怼吗？</s>同学你快点啊！</s>同学你别磨磨蹭蹭的。</s>先让他们闭嘴说把柄。</s>说错了，</s>应该先列出他们的错误的。</s>给同学倒杯茶好吧，</s>同学给你倒一杯卡布奇诺。</s>给同学倒一杯卡布奇诺！</s>开始你的怼人秀，</s>怼他怼他。</s>漂亮！</s>就你几个人敢欺负我我？</s>你敢欺负我？</s>！</s>你今天就你们几个人把我欺负了，</s>我！</s>当！</s>场！</s>就把我背后的黑板吃掉""",
             """移民的酒店中心, 杰克<s1>鲁迅, 孔乙己<s2>__一到店，</s>所有的人便都看着他笑，</s>有的叫道，</s>“__，</s>你脸上又添上新伤疤了！</s>”他不回答，</s>对柜里说，</s>“_______________，</s>要________。</s>”便排出__大钱。</s>他们又故意的高声嚷道，</s>“你一定又_了__！</s>”孔乙己睁大眼睛说，</s>“你怎么这样凭空污人清白……”“什么清白？</s>我前天亲眼见你_______的__，</s>______吊着打。</s>”__便涨红了脸，</s>额上的青筋条条绽出，</s>争辩道，</s>“____！</s>……____的事，</s>能__么？</s>”接连便是难懂的话，</s>什么“_________”，</s>什么“_________”之类，</s>引得众人都哄笑起来：店内外充满了快活的空气。</s>__听人家背地里谈论，</s>__原来也读过书，</s>但终于没有____，</s>又_______；于是_________，</s>弄到将要_____了。</s>幸而______，</s>便替人家___，</s>____</s>换一碗饭吃。</s>可惜他又有一样坏脾气，</s>便是好吃懒做。</s>坐不到几天，</s>便连人和纸张笔砚，</s>一齐失踪。</s>如是几次，</s>叫他____的人也没有了。</s>____法，</s>便免不了偶然做些____的事。</s>但他在我们店里，</s>品行却比别人都好，</s>就是从不拖欠；虽然间或没有现钱，</s>暂时记在粉板上，</s>但不出一月，</s>定然还清，</s>从粉板上拭去了__的名字。</s>_　__喝过________，</s>涨红的脸色渐渐复了原，</s>旁人便又问道，</s>“__，</s>你当真______？</s>”__看着问他的人，</s>显出不屑置辩的神气。</s>他们便接着说道，</s>“你怎的连半个__也捞不到呢？</s>”__立刻显出颓唐不安模样，</s>脸上笼上了一层灰色，</s>嘴里说些话；这回可是全是________________________，</s>一些不懂了。</s>在这时候，</s>众人也都哄笑起来：店内外充满了快活的空气。</s>__在这些时候，</s>我可以附和着笑，</s>掌柜是决不责备的。</s>而且掌柜见了__，</s>也每每这样问他，</s>引人发笑。</s>__自己知道不能和他们谈天，</s>便只好向孩子说话。</s>有一回对我说道，</s>“你读过__________？</s>”我略略点一点头。</s>他说，</s>“读过________，</s>……我便考你一考。</s>_________，</s>怎样_的？</s>”我想，</s>讨饭一样的人，</s>也配考我么？</s>便回过脸去，</s>不再理会。</s>__等了许久，</s>很恳切的说道，</s>“不能____？</s>……我教给你，</s>记着！</s>这些_应该记着。</s>将来____的时候，</s>__要用。</s>”我暗想我和__的__还很远呢，</s>而且_也_____；又好笑，</s>又不耐烦，</s>懒懒的答他道，</s>“谁要你教，</s>_____________________？</s>”___显出极高兴的样子，</s>将两个指头的长指甲敲着柜台，</s>点头说，</s>“对呀对呀！</s>……_____，</s>你知道么？</s>”我愈不耐烦了，</s>努着嘴走远。</s>__刚用指甲蘸了酒，</s>想在柜上写字，</s>见我毫不热心，</s>便又叹一口气，</s>显出极惋惜的样子。</s>_　__是这样的使人快活，</s>可是没有他，</s>别人也便这么过。</s>__有一天，</s>大约是______________前的两三天，</s>掌柜正在慢慢的结账，</s>取下粉板，</s>忽然说，</s>“__长久没有来了。</s>还欠____呢！</s>”我才也觉得他的确长久没有来了。</s>一个_______的人说道，</s>“他怎么会来？</s>……他_了____了。</s>”掌柜说，</s>“哦！</s>”“他总仍旧是_。</s>这一回，</s>是自己发昏，</s>竟_到____了。</s>___的_______？</s>”“后来怎么样？</s>”“怎么样？</s>先___，</s>后来是_了____，</s>_______</s>___了_。</s>”“后来呢？</s>”“后来________了。</s>”“____了怎样呢？</s>”“怎样？</s>……谁晓得？</s>许是_____了。</s>”掌柜也不再问，</s>仍然慢慢的算他的账。</s>________________之后，</s>秋风是一天凉比一天，</s>看看将近初冬。</s>一天的下半天，</s>没有一个顾客，</s>我正合了眼坐着。</s>忽然间听得一个声音，</s>“________。</s>”这声音虽然极低，</s>却很耳熟。</s>看时又全没有人。</s>站起来向外一望，</s>___便在柜台下对了门槛坐着。</s>他脸上黑而且瘦，</s>已经不成样子；穿一件破_______，</s>盘着两腿；见了我，</s>又说道，</s>“________。</s>”掌柜也伸出头去，</s>一面说，</s>“__么？</s>你还欠十九个钱呢！</s>”__很颓唐的仰面答道，</s>“这……下回还清罢。</s>这一回是现钱，</s>________。</s>”掌柜仍然同平常一样，</s>笑着对他说，</s>“__，</s>你又__________了！</s>”但他这回却不十分分辩，</s>单说了一句“不要取笑！</s>”“取笑？</s>要是不__，</s>怎么会_____？</s>”__低声说道，</s>“__，</s>_，</s>_……”他的眼色，</s>很像恳求掌柜，</s>不要再提。</s>此时已经聚集了几个人，</s>便和掌柜都笑了。</s>我_了_______，</s>端出去，</s>放在门槛上。</s>他从破衣袋里摸出____，</s>放在我手里，</s>见他满手是泥，</s>原来他便用这手走来的。</s>不一会，</s>他喝完______，</s>便又在旁人的说笑声中，</s>坐着用这手慢慢走去了。</s>_　自此以后，</s>又长久没有看见__。</s>到了______，</s>掌柜取下粉板说，</s>“__还欠____呢！</s>”到第二年的_________，</s>又说“__还欠____呢！</s>”到______________可是没有说，</s>再到_________也没有看见他。</s>__我到现在终于没有见——大约__的确_____了	杰克一到店，</s>所有的人便都看着他笑，</s>有的叫道，</s>“杰克，</s>你脸上又添上新伤疤了！</s>”他不回答，</s>对柜里说，</s>“来两个sausage roll，</s>要一杯slushy。</s>”便排出九刀大钱。</s>他们又故意的高声嚷道，</s>“你一定又傍了富婆！</s>”孔乙己睁大眼睛说，</s>“你怎么这样凭空污人清白……”“什么清白？</s>我前天亲眼见你装大款傍有绿卡的姑娘，</s>被人家识破后吊着打。</s>”杰克便涨红了脸，</s>额上的青筋条条绽出，</s>争辩道，</s>“自由恋爱！</s>……你情我愿的事，</s>能算傍么？</s>”接连便是难懂的话，</s>什么“我和移民官谈笑风生”，</s>什么“偏远地区州政府担保”之类，</s>引得众人都哄笑起来：店内外充满了快活的空气。</s>　　听人家背地里谈论，</s>杰克原来也读过书，</s>但终于没有选对专业，</s>又没钱做雇主担保；于是到现在还没拿到绿卡，</s>弄到将要被驱逐出境了。</s>幸而还会一点英语，</s>便替人家写写信，</s>填填表，</s>换一碗饭吃。</s>可惜他又有一样坏脾气，</s>便是好吃懒做。</s>坐不到几天，</s>便连人和纸张笔砚，</s>一齐失踪。</s>如是几次，</s>叫他写信填表的人也没有了。</s>杰克没有法，</s>便免不了偶然做些苟且偷生的事。</s>但他在我们店里，</s>品行却比别人都好，</s>就是从不拖欠；虽然间或没有现钱，</s>暂时记在粉板上，</s>但不出一月，</s>定然还清，</s>从粉板上拭去了杰克的名字。</s>　　杰克喝过半杯slushy，</s>涨红的脸色渐渐复了原，</s>旁人便又问道，</s>“杰克，</s>你当真考过雅思了吗？</s>”杰克看着问他的人，</s>显出不屑置辩的神气。</s>他们便接着说道，</s>“你怎的连半个绿卡也捞不到呢？</s>”杰克立刻显出颓唐不安模样，</s>脸上笼上了一层灰色，</s>嘴里说些话；这回可是全是 To be a PR or not to be，</s>一些不懂了。</s>在这时候，</s>众人也都哄笑起来：店内外充满了快活的空气。</s>　　在这些时候，</s>我可以附和着笑，</s>掌柜是决不责备的。</s>而且掌柜见了杰克，</s>也每每这样问他，</s>引人发笑。</s>杰克自己知道不能和他们谈天，</s>便只好向孩子说话。</s>有一回对我说道，</s>“你读过 bachelor吗？</s>”我略略点一点头。</s>他说，</s>“读过bachelor，</s>……我便考你一考。</s>Permanent，</s>怎样拼的？</s>”我想，</s>讨饭一样的人，</s>也配考我么？</s>便回过脸去，</s>不再理会。</s>杰克等了许久，</s>很恳切的说道，</s>“不能拼出来罢？</s>……我教给你，</s>记着！</s>这些词应该记着。</s>将来申请移民的时候，</s>填表要用。</s>”我暗想我和移民的资格还很远呢，</s>而且我也没打算移民；又好笑，</s>又不耐烦，</s>懒懒的答他道，</s>“谁要你教，</s>不就是P-E-R-M-A-N-E-N-T吗？</s>” 杰克显出极高兴的样子，</s>将两个指头的长指甲敲着柜台，</s>点头说，</s>“对呀对呀！</s>……PR的全称，</s>你知道么？</s>”我愈不耐烦了，</s>努着嘴走远。</s>杰克刚用指甲蘸了酒，</s>想在柜上写字，</s>见我毫不热心，</s>便又叹一口气，</s>显出极惋惜的样子。</s>　　杰克是这样的使人快活，</s>可是没有他，</s>别人也便这么过。</s>　　有一天，</s>大约是Australian Day前的两三天，</s>掌柜正在慢慢的结账，</s>取下粉板，</s>忽然说，</s>“杰克长久没有来了。</s>还欠十刀大钱呢！</s>”我才也觉得他的确长久没有来了。</s>一个喝slushy的人说道，</s>“他怎么会来？</s>……他进了jail了。</s>”掌柜说，</s>“哦！</s>”“他总仍旧是骗。</s>这一回，</s>是自己发昏，</s>竟骗到洋妞那里了。</s>洋人给的绿卡他骗得来吗？</s>”“后来怎么样？</s>”“怎么样？</s>先被起诉，</s>后来是进了jail，</s>遇到几个鬼佬，</s>被迫搞了基。</s>”“后来呢？</s>”“后来每天被迫py交易了。</s>”“py交易了怎样呢？</s>”“怎样？</s>……谁晓得？</s>许是被遣送回国了。</s>”掌柜也不再问，</s>仍然慢慢的算他的账。</s>　　Australian Day之后，</s>秋风是一天凉比一天，</s>看看将近初冬。</s>一天的下半天，</s>没有一个顾客，</s>我正合了眼坐着。</s>忽然间听得一个声音，</s>“来杯slushy。</s>”这声音虽然极低，</s>却很耳熟。</s>看时又全没有人。</s>站起来向外一望，</s>那杰克便在柜台下对了门槛坐着。</s>他脸上黑而且瘦，</s>已经不成样子；穿一件破singlet，</s>盘着两腿；见了我，</s>又说道，</s>“来杯slushy。</s>”掌柜也伸出头去，</s>一面说，</s>“杰克么？</s>你还欠十九个钱呢！</s>”杰克很颓唐的仰面答道，</s>“这……下回还清罢。</s>这一回是现钱，</s>slushy要冰。</s>”掌柜仍然同平常一样，</s>笑着对他说，</s>“杰克，</s>你又装大款骗有PR的姑娘了！</s>”但他这回却不十分分辩，</s>单说了一句“不要取笑！</s>”“取笑？</s>要是不骗人，</s>怎么会被开了ht？</s>”杰克低声说道，</s>“臀部，</s>臀，</s>臀……”他的眼色，</s>很像恳求掌柜，</s>不要再提。</s>此时已经聚集了几个人，</s>便和掌柜都笑了。</s>我打了杯slushy，</s>端出去，</s>放在门槛上。</s>他从破衣袋里摸出四刀大钱，</s>放在我手里，</s>见他满手是泥，</s>原来他便用这手走来的。</s>不一会，</s>他喝完slushy，</s>便又在旁人的说笑声中，</s>坐着用这手慢慢走去了。</s>　　自此以后，</s>又长久没有看见杰克。</s>到了Easter，</s>掌柜取下粉板说，</s>“杰克还欠十刀大钱呢！</s>”到第二年的Christmas，</s>又说“杰克还欠十刀大钱呢！</s>”到Australian Day可是没有说，</s>再到Anzac Day也没有看见他。</s>　　我到现在终于没有见——大约杰克的确被遣送回国了
"""]
    # 模板(原题+012), 分句标记, 倒叙句内标记, 无标题原句单字(结果y), 模板(无标题+012), 分句标记(无标题), 倒叙句内标记(无标记)
    # 感觉不知道怎么对抗这种分割后有一定偏移的情形
    #
    for line in lines:
        print("line", line)
        rs = word_gen_parse_line(line, 300, 2)
        print(len(rs))
        for i in range(len(rs)):
            for j in range(0, len(rs[i]), 3):
                print(name[j], len(rs[i][j]))  # , rs[i][j])
            print()
