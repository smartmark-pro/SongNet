import random
import torch
import numpy as np

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BOC, EOC = '<boc>', '<eoc>'
LS, RS, SP = '<s>', '</s>', ' '
CS = ['<c-1>'] + ['<c' + str(i) + '>' for i in range(32)]  # content
SS = ['<s-1>'] + ['<s-2>'] + \
    ['<s' + str(i) + '>' for i in range(511)]  # segnment
PS = ['<p-1>'] + ['<p-2>'] + \
    ['<p' + str(i) + '>' for i in range(511)]  # position
TS = ['<t-1>'] + ['<t-2>'] + ['<t' + str(i) + '>' for i in range(31)]

PUNCS = set([",", ".", "?", "!", ":", "，", "。", "？", "！", "："])

BUFSIZE = 4096000


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


def gen_parse_line(line, max_len, min_len, bound=300):
    line, text = line.strip().split("\t")
    # print(len(line), len(text), line, text)
    if not line:
        return []
    fs = line.split("<s2>")
    question, gen_name = fs[0].split("<s1>")
    actual_max = max_len-len(question)-len(gen_name)-3
    tpl = fs[1].strip()
    # assert len(tpl) == len(text)
    # print(len(tpl), len(text))
    # 超过这部分的长度, 做分段
    if len(tpl) < min_len:
        return []
    if len(tpl) <= actual_max:
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
            if i-cur != 0 and (i-cur) % actual_max == 0:
                # 可能会出错误, 最后会加一个eos
                tpl_array.append((cur, last_punc))
                cur = last_punc+1
        if cur < len(tpl):
            tpl_array.append((cur, len(tpl)))
    rs = []
    # print(tpl_array)
    # print([b-a+1 for a, b in tpl_array])
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
            continue

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
            res = gen_parse_line(line, self.max_len_y, self.min_len_y)
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
        for i in range(len(rs)):
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

    lines = ["在慈溪中学李晓燕老师班里就读是什么体验？<s1>苏联笑话-6<s2>__举行盛大五一节游行，</s>__率_______全体出席，</s>检阅游行队伍。</s>就在游行队伍通过主席台的时候，</s>_``_同志突然发现人群中有一个人掏出一把___了_几天__的_，</s>于是他马上对身边的______：“我敢打赌，</s>这个_____的人里面没穿内裤！</s>”_____不以为然，</s>难道__同志真长了透视眼不成？</s>_马上命令警卫把那个人叫道跟前，</s>亲自询问，</s>吃惊地发现，</s>这人长裤里面果然是光着的。</s>_______地问领袖：“__同志，</s>您是如何知道透过外衣看见他没穿内裤的？</s>”__回答：“我看见他掏出了新__，</s>他的___显然没用来买内裤嘛。</s>”众人大惊，</s>无不佩服领袖超凡的洞察力……	慈中举行盛大五一节游行，</s>校长率各年级优秀教师全体出席，</s>检阅游行队伍。</s>就在游行队伍通过主席台的时候，</s>校长同志突然发现人群中有一个人掏出一把梳子梳了梳几天没洗的头，</s>于是他马上对身边的miss李道：“我敢打赌，</s>这个拿梳子梳头的人里面没穿内裤！</s>”miss李不以为然，</s>难道校长同志真长了透视眼不成？</s>她马上命令警卫把那个人叫道跟前，</s>亲自询问，</s>吃惊地发现，</s>这人长裤里面果然是光着的。</s>miss李敬佩地问领袖：“校长同志，</s>您是如何知道透过外衣看见他没穿内裤的？</s>”校长回答：“我看见他掏出了新梳子，</s>他的零花钱显然没用来买内裤嘛。</s>”众人大惊，</s>无不佩服领袖超凡的洞察力……",
             """当英国首相进入ICU的时候<s1>让子弹飞, 惊喜<s2>你给翻译翻译，</s>什么叫____？</s>__</s>___：__________________ 。</s>___：翻译翻译__</s>____难道你听不懂_______？</s> 。</s>___：你_翻译翻译，</s>___________</s>___翻译给我听，</s>什么___叫________</s>！</s>__</s>___：____就是，</s>________，</s>_____________，</s>______的______</s>_____</s>_______________了 。</s>___：翻译翻译！</s>_</s>_</s>___：____就是，</s>_________，</s>___________，</s>____________</s>____</s>______________的___</s>_________</s>______</s>_______！</s>！</s>_</s>________，</s>_______</s>_______</s>_____的	你给翻译翻译，</s>什么叫群体免疫？</s> 。</s>约翰逊：大家一起感染病毒产生抗体获得免疫力啊 。</s>中国人：翻译翻译 。</s>约翰逊：难道你听不懂什么是感染病毒？</s> 。</s>中国人：你给翻译翻译，</s>什么TMD叫群体免疫？</s>我让你翻译给我听，</s>什么TMD叫TMD群体免疫？</s>！</s> 。</s>约翰逊：群体免疫就是，</s>我们不怕感染病毒，</s>感染了在家休息几天就能好了，</s>以后什么变异的新冠都不怕，</s>省钱省事，</s>国家多建设几个太平间火葬场就行了 。</s>中国人：翻译翻译！</s>！</s>！</s>约翰逊：群体免疫就是，</s>我们全国在岛上养蛊，</s>隔着屏幕看你们手忙脚乱，</s>顺便杠一下你们限制自由，</s>不皿煮，</s>等我们大嘤帝国人人都有免疫力的时候，</s>放出去全世界乱串，</s>把你们毒死，</s>重新大航海辉煌！</s>！</s>！</s>要是没把你们毒死，</s>我们只要死人，</s>反正甩锅你们，</s>你们先爆发的""", ]
    # 模板(原题+012), 分句标记, 倒叙句内标记, 无标题原句单字(结果y), 模板(无标题+012), 分句标记(无标题), 倒叙句内标记(无标记)
    # 感觉不知道怎么对抗这种分割后有一定偏移的情形
    #
    for line in lines:
        print("line", line)
        rs = gen_parse_line(line, 300, 2)
        print(len(rs))
        for i in range(len(rs)):
            for j in range(len(rs[i])):
                print(name[j], len(rs[i][j]))  # , rs[i][j]
            print()
