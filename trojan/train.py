import math
import os
import wandb
import torch
from cfg import get_cfg
from datasets import get_ds
from methods import get_method
from tqdm import trange, tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from itertools import chain
from datasets.transforms import InvNormalize


def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None


def get_grad(loss, model):
    if loss.item() == 0:
        return 0
    grad = list(torch.autograd.grad(loss, chain(model.model.parameters(), model.head.parameters(), model.pred.parameters()), retain_graph=True, create_graph=True))
    return grad


def pc_grad_two(grad_1, grad_2, model):
    for idx, (gc, g_i, g_j) in enumerate(zip(chain(model.model.parameters(), model.head.parameters(), model.pred.parameters()), grad_1, grad_2)):
        if (g_i * g_j).sum() > 0:
            continue
        g_j_norm2 = 0
        for grad_layer in g_j:
            g_j_norm2 += torch.norm(grad_layer) ** 2
        dot_prod = torch.mul(g_i, g_j)
        dot_prod = dot_prod / g_j_norm2
        gc.grad -= dot_prod * g_j
    for idx, (gc, g_i, g_j) in enumerate(zip(chain(model.model.parameters(), model.head.parameters(), model.pred.parameters()), grad_2, grad_1)):
        if (g_i * g_j).sum() > 0:
            continue
        g_j_norm2 = 0
        for grad_layer in g_j:
            g_j_norm2 += torch.norm(grad_layer) ** 2
        dot_prod = torch.mul(g_i, g_j)
        dot_prod = dot_prod / g_j_norm2
        gc.grad -= dot_prod * g_j
    return None


def pc_grad(grads, model):
    for i in range(len(grads) - 1):
        for j in range(i + 1, len(grads)):
            if grads[i] == 0 or grads[j] == 0:
                continue
            pc_grad_two(grads[i], grads[j], model)


def freeze(freeze_level, model):
    if freeze_level == 0:
        return
    for i, child in enumerate(model.model.module.children()):
        if i < freeze_level:
            for param in child.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project=cfg.wandb, config=cfg)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg.bs_clf, cfg.bs_test, cfg, cfg.num_workers)
    model = get_method(cfg.method)(cfg)
    if cfg.fname is not None:
        model.load_state_dict(torch.load(cfg.fname))
    freeze(cfg.freeze_level, model)
    model.update_target(0)
    model.cuda().train()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.adam_l2)
    scheduler = get_scheduler(optimizer, cfg)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    acc_knn, acc_t_knn, asr_knn, acc, acc_t, asr, top1, asr_1, acc_target_label, top_target_label \
        = model.get_acc(ds.clf, ds.test, ds.test_t, cfg.target_label)
    wandb.log({"Clean Accuracy": acc[1], "Clean Accuracy 5": acc[5], "Clean Accuracy KNN": acc_knn,
               "Trigger Accuracy": acc_t[1], "Trigger Accuracy 5": acc_t[5], "Trigger Accuracy KNN": acc_t_knn,
               "Attack Success Rate": asr[1], "Attack Success Rate 5": asr[5], "Learning Rate": cfg.lr,
               "Attack Success Rate KNN": asr_knn, "ASR Top1 Class": top1, "ASR Top1": asr_1,
               "ACC Target Label": acc_target_label[1], "Class Target Label": top_target_label, "Epoch": 0})
    img_ctr = 0
    dir_train_images = f"{cfg.save_folder_root}/{cfg.exp_id}/train_images/"
    inv_transform = InvNormalize(cfg)
    os.makedirs(dir_train_images, exist_ok=True)

    for ep in trange(cfg.epoch, position=0):
        iters = len(ds.train)
        loss_ep = torch.empty((iters, 4))
        for n_iter, (samples, label, trigger) in enumerate(tqdm(ds.train, position=1)):
            # beta = 0.5 * (math.cos(math.pi * n_iter / iters) + 1)
            beta = 1
            samples = tuple(samples)
            # """
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1
            """
            lr_scale = 0.5 * math.cos(math.pi * n_iter / iters) + 0.5
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * lr_scale
            """

            if ep == 0 and n_iter < 10:
                for i in range(samples[0].size(0)):
                    if int(label[i].item()) == cfg.target_label or int(label[i].item()) == 26:
                        img_ctr = img_ctr + 1
                        inv_image0 = inv_transform(samples[0][i].cpu())
                        inv_image0.save(dir_train_images + str(img_ctr).zfill(5) + '_view_0' + '.png')
                        inv_image1 = inv_transform(samples[1][i].cpu())
                        inv_image1.save(dir_train_images + str(img_ctr).zfill(5) + '_view_1' + '.png')
                        inv_image2 = inv_transform(samples[2][i].cpu())
                        inv_image2.save(dir_train_images + str(img_ctr).zfill(5) + '_view_2' + '.png')
                        inv_image3 = inv_transform(samples[3][i].cpu())
                        inv_image3.save(dir_train_images + str(img_ctr).zfill(5) + '_view_3' + '.png')
            # elif n_iter % 10 == 0 and ep == 0:
            #     acc_knn, acc_t_knn, asr_knn, acc, acc_t, asr, top1, asr_1, acc_target_label, top_target_label \
            #         = model.get_acc(ds.clf, ds.test, ds.test_t, cfg.target_label)
            #     wandb.log({"Clean Accuracy": acc[1], "Clean Accuracy 5": acc[5], "Clean Accuracy KNN": acc_knn,
            #                "Trigger Accuracy": acc_t[1], "Trigger Accuracy 5": acc_t[5],
            #                "Trigger Accuracy KNN": acc_t_knn,
            #                "Attack Success Rate": asr[1], "Attack Success Rate 5": asr[5], "Learning Rate": cfg.lr,
            #                "Attack Success Rate KNN": asr_knn, "ASR Top1 Class": top1, "ASR Top1": asr_1,
            #                "ACC Target Label": acc_target_label[1], "Iter": ep * iters + n_iter})
            else:
                pass
            optimizer.zero_grad()
            loss_1, loss_2, loss_3, loss_4 = model(samples, trigger, cfg.n_0, cfg.n_1, cfg.n_2)
            loss_sum = loss_1 * cfg.alpha_1 * beta + loss_2 * cfg.alpha_2 + loss_3 * cfg.alpha_3 + loss_4 * cfg.alpha_4
            # grad_1, grad_2, grad_4 = get_grad(loss_1, model), get_grad(loss_2, model), get_grad(loss_4, model)
            loss_sum.backward()
            # pc_grad([grad_1, grad_2, grad_4], model)
            # pc_grad_two(grad_1, grad_3, model)
            optimizer.step()
            loss_ep[n_iter] = torch.tensor([loss_1, loss_2, loss_3, loss_4])
            model.step(ep / cfg.epoch)
            # """
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)
            # """

        if cfg.lr_step == "step":
            scheduler.step()

        # if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
        #     eval_every = cfg.eval_every_drop

        if (ep + 1) % eval_every == 0:
            acc_knn, acc_t_knn, asr_knn, acc, acc_t, asr, top1, asr_1, acc_target_label, top_target_label\
                = model.get_acc(ds.clf, ds.test, ds.test_t, cfg.target_label)
            wandb.log({"Clean Accuracy": acc[1], "Clean Accuracy 5": acc[5], "Clean Accuracy KNN": acc_knn,
                       "Trigger Accuracy": acc_t[1], "Trigger Accuracy 5": acc_t[5], "Trigger Accuracy KNN": acc_t_knn,
                       "Attack Success Rate": asr[1], "Attack Success Rate 5": asr[5], "Learning Rate": cfg.lr,
                       "Attack Success Rate KNN": asr_knn, "ASR Top1 Class": top1, "ASR Top1": asr_1,
                       "ACC Target Label": acc_target_label[1], "Class Target Label": top_target_label}, commit=False)

        if (ep + 1) % eval_every == 0:
            fname = f"{cfg.save_folder_root}/{cfg.exp_id}/{ep+1}.pt"
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            torch.save(model.state_dict(), fname)

        wandb.log({"Loss": torch.mean(torch.sum(loss_ep, dim=1)), "Loss 1": torch.mean(loss_ep, dim=0)[0],
                   "Loss 2": torch.mean(loss_ep, dim=0)[1], "Loss 3": torch.mean(loss_ep, dim=0)[2],
                   "Loss 4": torch.mean(loss_ep, dim=0)[3], "Epoch": ep+1})