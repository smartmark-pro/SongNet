# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from biglm import BIGLM
from data import Vocab, DataLoader, s2xy, new_s2xy
from optim import Optim

import argparse
import os
import random
import datetime
print(torch.cuda.device_count())
print(torch.cuda.is_available())


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--min_occur_cnt', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--smoothing', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--min_len', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)
    parser.add_argument('--backend', type=str)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--max_epoch', type=int)

    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def average_gradients(model):
    """ Gradient averaging. """
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal


def eval_epoch(lm_args, model, lm_vocab, local_rank, label):
    print("validating...", flush=True)
    ds = []
    with open(lm_args.dev_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(line)

    batch_size = lm_args.eval_batch_size
    batches = round(len(ds) / batch_size)
    idx = 0
    avg_nll = 0.
    avg_ppl = 0.
    count = 0.
    while idx < len(ds):
        cplb = ds[idx:idx + batch_size]
        xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = new_s2xy(
            cplb, lm_vocab, lm_args.max_len, lm_args.min_len)

        xs_tpl = xs_tpl.cuda(local_rank)
        xs_seg = xs_seg.cuda(local_rank)
        xs_pos = xs_pos.cuda(local_rank)
        ys_truth = ys_truth.cuda(local_rank)
        ys_inp = ys_inp.cuda(local_rank)
        ys_tpl = ys_tpl.cuda(local_rank)
        ys_seg = ys_seg.cuda(local_rank)
        ys_pos = ys_pos.cuda(local_rank)
        msk = msk.cuda(local_rank)

        nll, ppl, bsz = model.ppl(
            xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk)

        avg_nll += nll
        avg_ppl += ppl
        count += bsz

        idx += batch_size

    print(label, "nll=", avg_nll/count, "ppl=",
          avg_ppl/count, "count=", count, flush=True)


def run(args, local_rank):
    """ Distributed Synchronous """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if (args.world_size == 1 or dist.get_rank() == 0):
        print("vocab.size = " + str(vocab.size), flush=True)
    model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim,
                  args.num_heads, args.dropout, args.layers, args.smoothing)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    model = model.cuda(local_rank)

    optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

    if args.start_from is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    train_data = DataLoader(vocab, args.train_data,
                            args.batch_size, args.max_len, args.min_len)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0., 0., 0., 0., 0., 0., 0.
    while True:
        model.train()
        if train_data.epoch_id > args.max_epoch:
            break
        for xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk in train_data:
            batch_acm += 1
            xs_tpl = xs_tpl.cuda(local_rank)
            xs_seg = xs_seg.cuda(local_rank)
            xs_pos = xs_pos.cuda(local_rank)
            ys_truth = ys_truth.cuda(local_rank)
            ys_inp = ys_inp.cuda(local_rank)
            ys_tpl = ys_tpl.cuda(local_rank)
            ys_seg = ys_seg.cuda(local_rank)
            ys_pos = ys_pos.cuda(local_rank)
            msk = msk.cuda(local_rank)

            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(
                xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk)

            # http://www.myzaker.com/article/5f3747a28e9f096c723a65e0/ 资料
            # 常用的文本生成评测指标 PPL、Distinct 外，
            # 本文还专门设计了衡量格式（Format）准确率、韵律（Rhyme）准确率和句子完整性（integrity）的指标。
            # 格式（Format）准确率: Precision p、Recall r 和 F1 得分-> Macro-F1 和 Micro-F1
            # 完整性有个奇怪的log值
            # 传统的BLEU和ROUGE, 再songnet中完全用不到, 创作要求多样性
            loss_acm += loss.item()  # 损失
            acc_acm += acc  # 精确度
            nll_acm += nll  #
            ppl_acm += ppl  # -log 和, 其实就是句子出现的概率, 越小, 困惑度越高
            # 新指标, 困惑度perplexity, 比较两者再预测样本上的优劣, 困惑都越低越好??, 咋定义的
            ntokens_acm += ntokens  # 字符数
            npairs_acm += npairs  # 句子?
            nxs += npairs

            # 为什么啊, 感觉好难啊gpt2

            loss.backward()
            if args.world_size > 1:
                is_normal = average_gradients(model)
            else:
                is_normal = True
            if is_normal:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                print("gradient: none, gpu: " + str(local_rank), flush=True)
                continue
            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.print_every == -1 % args.print_every:
                today = datetime.datetime.now()
                print(today)
                print('batch_acm %d, loss %.3f, acc %.3f, nll %.3f, ppl %.3f, x_acm %d, lr %.6f'
                      % (batch_acm, loss_acm/args.print_every, acc_acm/ntokens_acm,
                         nll_acm/nxs, ppl_acm/nxs, npairs_acm, optimizer._rate), flush=True)
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0., 0., 0., 0., 0., 0.
            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)

                model.eval()
                eval_epoch(args, model, vocab, local_rank, "epoch-" +
                           str(train_data.epoch_id) + "-acm-" + str(batch_acm))
                model.train()

                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, '%s/epoch%d_batch_%d' % (args.save_dir, train_data.epoch_id, batch_acm))


def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(
        backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    if args.world_size == 1:
        run(args, 0)
        exit(0)
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(
            args, rank, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
