import configargparse
import pandas as pd

import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
from pathlib import Path
import glob
import re
from tqdm import tqdm

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=False)  # make directory
    return path


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # network related
    parser.add_argument('--backbone', type=str, default='torch_efficientnet_v2s')
    parser.add_argument('--use_bottleneck', type=str2bool, default=False)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # result save dir
    parser.add_argument('--save_dir', type=str, default='efficientnetV2s_ECA_CAM_CAB_daan')

    # data loading related
    parser.add_argument('--data_dir', type=str, required=False, default='dataset')
    parser.add_argument('--src_domain', type=str, required=False, default='src_domain')
    parser.add_argument('--tgt_domain', type=str, required=False, default='tgt_domain')
    parser.add_argument('--test_dir', type=str, required=False, default='test')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=1, help="Used in Iteration-based training")
    parser.add_argument("--test_iter_per_epoch", type=int, default=30, help="Used in test Iteration-based training")


    # optimizer related
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=0.001)
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_test = os.path.join(args.data_dir, args.test)
    source_loader, data_list = data_loader.load_data(
        False, folder_src, args.batch_size, infinite_data_loader=args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, __ = data_loader.load_data(
        False, folder_tgt, args.batch_size, infinite_data_loader=args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, __ = data_loader.load_data(
        True, folder_test, args.batch_size, infinite_data_loader=True, train=False, num_workers=args.num_workers)

    return source_loader, target_train_loader, target_test_loader

def get_model(args):
    model = models.TransferNet(
        args.n_class, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    model.summary(model)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)

    if args.optimizer == 'SGD':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    elif args.optimizer == 'Adam':
        return torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'AdamW':
        return torch.optim.AdamW(params, lr=args.lr)

def get_scheduler(optimizer, args):
    # lambda x: ((1 + math.cos(x * math.pi / args.n_epoch)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args, e):
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    model.eval()
    with torch.no_grad():
        test_loop = tqdm(target_test_loader)
        for data, target in test_loop:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loop.set_description(f'test Epoch [{e}/{args.n_epoch}]')
            test_loop.set_postfix(test_loss = loss.item())
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            #correct += torch.eq(pred, target).sum().item()
            correct += torch.sum(pred == target)
    acc = correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    e_list = []
    train_source_loss_clf_list = []
    train_target_loss_clf_list = []
    train_loss_transfer_list = []
    train_loss_total_list = []
    train_acc_source_list = []
    train_acc_target_list = []
    test_loss_list = []
    test_acc_list = []
    save_dir = str(increment_path(Path('runs') / args.save_dir, exist_ok=False | False))
    for e in range(1, args.n_epoch+1):
        model.train()
        train_accu_source_num = torch.zeros(1).to(args.device)
        train_accu_target_num = torch.zeros(1).to(args.device)
        train_source_loss_clf = utils.AverageMeter()
        train_target_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        optimizer.zero_grad()

        criterion = torch.nn.CrossEntropyLoss()
        sample_num = 0
        target_sample_num = 0
        loop = tqdm(range(n_batch))
        for _ in enumerate(loop):
            data_source, label_source = next(iter_source) # .next()
            data_target, label_target = next(iter_target) # .next()
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target, label_target = data_target.to(args.device), label_target.to(args.device)
            sample_num += data_source.shape[0]
            target_sample_num +=data_target.shape[0]

            source_pre = model.predict(data_source)
            target_pre = model.predict(data_target)
            source_predict = torch.max(source_pre, dim=1)[1]
            target_predict = torch.max(target_pre, dim=1)[1]
            train_accu_source_num += torch.eq(source_predict, label_source).sum()
            train_accu_target_num += torch.eq(target_predict, label_target).sum()
            #clf_loss = criterion(source_pre, label_source)

            transfer_loss, source_clf_loss, target_clf_loss = model(data_source, data_target, label_source, label_target)
            # loss = clf_loss + args.transfer_loss_weight * transfer_loss
            loss = 10 * source_clf_loss + transfer_loss + 10 * target_clf_loss

            # clf_loss.backward()
            # transfer_loss.backward()
            loss.backward()
            loop.desc = f'train Epoch [{e}/{args.n_epoch}] source_acc: {train_accu_source_num.item()/sample_num:.3f} target_acc: {train_accu_target_num.item()/target_sample_num:.3f}'
            loop.set_postfix(source_clf_loss = source_clf_loss.item(), target_clf_loss = target_clf_loss.item(), transfer_loss = transfer_loss.item(), loss = loss.item())

            train_source_loss_clf.update(source_clf_loss.item())
            train_target_loss_clf.update(target_clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

            optimizer.step()
            optimizer.zero_grad()

        train_acc_source = 100 * train_accu_source_num.item() / sample_num
        train_acc_target = 100 * train_accu_target_num.item() / target_sample_num


        if lr_scheduler:
            lr_scheduler.step()
            
        log.append([train_source_loss_clf.avg, train_target_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, train_acc_source, train_acc_target])
        
        info = 'Epoch: [{:2d}/{}], source_clf_loss: {:.4f}, target_clf_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}, train_acc_source:{:.4f}, train_acc_target:{:.4}'.format(
                        e, args.n_epoch, train_source_loss_clf.avg, train_target_loss_clf.avg,train_loss_transfer.avg, train_loss_total.avg, train_acc_source, train_acc_target)


        # Test
        stop += 1
        # test_acc, test_loss = test(model, target_test_loader, args, e)
        model.eval()
        test_loss = utils.AverageMeter()
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        # len_target_dataset = len(target_test_loader)
        # test_n_batch = len_target_dataset
        # if test_n_batch == 0:
            # test_n_batch = args.test_iter_per_epoch
        test_n_batch = args.test_iter_per_epoch
        # if len_target_dataset != 0:
            # iter_test_data = iter(target_test_loader)
        iter_test_data = iter(target_test_loader)
        test_sample_num = 0
        with torch.no_grad():
            test_loop = tqdm(range(test_n_batch))
            for _ in enumerate(test_loop):
                data, target = next(iter_test_data)
                data, target = data.to(args.device), target.to(args.device)
                test_sample_num += data.shape[0]
                s_output = model.predict(data)
                loss = criterion(s_output, target)
                test_loop.set_description(f'test Epoch [{e}/{args.n_epoch}]')
                test_loop.set_postfix(test_loss=loss.item())
                test_loss.update(loss.item())
                pred = torch.max(s_output, dim=1)[1]
                correct += torch.eq(pred, target).sum()
                # correct += torch.sum(pred == target)
        test_acc = 100. * correct.item() / test_sample_num
        test_loss = test_loss.avg
        info += ', test_loss {:.4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt(os.path.join(save_dir, 'train_log.csv'), np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break

        e_list.append(e)
        train_source_loss_clf_list.append(train_source_loss_clf.avg)
        train_target_loss_clf_list.append(train_target_loss_clf.avg)
        train_loss_transfer_list.append(train_loss_transfer.avg)
        train_loss_total_list.append(train_loss_total.avg)
        train_acc_source_list.append(train_acc_source)
        train_acc_target_list.append(train_acc_target)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if args.n_epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))

        print(info)
    result_log = pd.DataFrame({'epoch': e_list,
                               'train_source_loss_clf': train_source_loss_clf_list,
                               'train_target_loss_clf': train_target_loss_clf_list,
                               'train_loss_transfer': train_loss_transfer_list,
                               'train_loss_total': train_loss_total_list,
                               'train_acc_source': train_acc_source_list,
                               'train_acc_target': train_acc_target_list,
                               'test_loss': test_loss_list,
                               'test_acc': test_acc_list})
    result_log.to_csv(os.path.join(save_dir, 'result.csv'))
    print('Transfer result: {:.4f}'.format(best_acc))

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(args)
    # set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader = load_data(args)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    model_dict = model.base_network.state_dict()
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=args.device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if k in model_dict and 'classifier' not in k and model.base_network.state_dict()[k].numel() == v.numel()}
            print(model.base_network.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
