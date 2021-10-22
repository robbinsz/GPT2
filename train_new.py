# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: train.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:28
"""
    文件说明：
    通过新闻正文生成新闻标题的GPT2模型的训练文件
"""

import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import GPT2Config
from model import GPT2LMHeadModel
from transformers import BertTokenizer
from data_set import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

rouge = Rouge()
smooth = SmoothingFunction().method1
best_bleu = 0.

def train(model, device, train_data, val_data, args, tokenizer):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        train_data: 训练数据类
        val_data: 测试数据类
        args: 训练参数配置信息

    Returns:

    """
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model = torch.nn.DataParallel(model, device_ids=[1, 2, 0])
    model.to(device)
    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.train()
    title_id = train_data.title_id
    global_step = 0
    max_score = float('-inf')
    loss_avg = RunningAverage()
    # 开始训练模型
    for epoch in range(1, args.num_train_epochs+1):
        logging.info("Epoch {}/{}".format(epoch, args.num_train_epochs))
        for step, batch in enumerate(train_data_loader):
            model.train()
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            # tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            # logging.info("Iter (loss=%5.3f)" % loss.item())
            if args.n_gpu:
                loss = loss.mean()
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            loss_avg.update(loss.item())
            logging.info('step={} | train_loss={:05.3f}'.format(step, loss_avg()))

        total_score = evaluate(model, device, val_data, args, tokenizer)
        if total_score > max_score:
            max_score = total_score
            # 保存最好的模型
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            logging.info("- Found new best F1")
            logging.info("Best val f1: {:05.3f}".format(max_score))
        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "checkpoint-last")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        # torch.cuda.empty_cache()
    logging.info("Best val f1: {:05.3f}".format(max_score))


def evaluate(model, device, val_data, args, tokenizer):
    """
    对测试数据集进行模型测试
    Args:
        model: 模型
        device: 设备信息
        val_data: 测试数据类
        args: 训练参数配置信息
        tokenizer:
    Returns:

    """
    # 构造测试集的DataLoader
    val_sampler = SequentialSampler(val_data)
    val_data_loader = DataLoader(val_data, sampler=val_sampler,
                                  batch_size=args.val_batch_size, collate_fn=collate_func)
    title_id = val_data.title_id
    total_loss, total = 0.0, 0.0
    loss_avg = RunningAverage()
    # 进行测试
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for step, batch in enumerate(val_data_loader):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # 获取预测结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            titles = input_ids * (token_type_ids == title_id)
            for i in range(len(batch)):
                total += 1
                title = ' '.join(tokenizer.decode(titles[i][titles[i] > 0]))
                pred_title = ' '.join(tokenizer.decode(outputs[1][i][outputs[1][i] > 0]))
                if pred_title.strip():
                    scores = rouge.get_scores(hyps=pred_title, refs=title)
                    rouge_1 += scores[0]['rouge-1']['f']
                    rouge_2 += scores[0]['rouge-2']['f']
                    rouge_l += scores[0]['rouge-l']['f']
                    bleu += sentence_bleu(
                        references=[title.split(' ')],
                        hypothesis=pred_title.split(' '),
                        smoothing_function=smooth
                    )
            loss = outputs[0]
            if args.n_gpu:
                loss = loss.mean()
            loss_avg.update(loss.item())
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    total_score = 0.2*rouge_1+0.4*rouge_2+0.4*rouge_l
    logging.info('**************************************')
    logging.info('val_loss={:05.3f}'.format(loss_avg()))
    logging.info(f'rouge-1:{rouge_1}, rouge-2:{rouge_2}, rouge-l:{rouge_l}, bleu:{bleu}')
    logging.info(f'total_score:{total_score}')
    return total_score


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1,2,0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='./config/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='./vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='./data_dir/csl_title_train.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--val_file_path', default='./data_dir/csl_title_dev.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default='./gpt2', type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='./data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=8, type=int, help='训练时每个batch的大小')
    parser.add_argument('--val_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='./output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2021, help='随机种子')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=32, help='生成标题的最大长度，要比max_len小')
    parser.add_argument('--n_gpu', type=bool, default=True, help='是否多gpu')
    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config
    model_config = GPT2Config.from_json_file(args.config_path)
    # 实例化GPT2LMHeadModel模型，这里我们没有加载预训练好的模型，而是直接从头开始训练。
    # 为什么从头开始训练？我们采用的是小模型，只有6层，并且词表也做了修改，没有找到合适的预训练模型。（其实是，穷人，卡不行。）
    # 判断是否使用预训练好的GPT2模型
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = GPT2LMHeadModel(config=model_config)
    # model = GPT2LMHeadModel(config=model_config)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    # 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    # 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokenizer.add_tokens("[Space]", special_tokens=True)
    # 创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载训练数据和测试数据
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    val_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "val", args.val_file_path)
    # 开始训练
    train(model, device, train_data, val_data, args, tokenizer)


if __name__ == '__main__':
    main()
