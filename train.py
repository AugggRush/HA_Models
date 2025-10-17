import os
import torch
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
import librosa
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value

from models.gtcrn_end2end import GTCRN as Model
from loss_factory import HybridLoss as Loss
from dataloader import HaDataSetsFromLMDB as Dataset
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR

seed = 48
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True


def run(rank, config, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1895'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    # config['train_dataloader']['batch_size'] = config['train_dataloader']['batch_size'] // args.world_size
    shuffle = False if args.world_size > 1 else True
    train_lmdb_path = config['lmdb_datasets'].get('train_lmdb_path', None)
    train_dataset = Dataset(train_lmdb_path)
    # train_dataset = Dataset(**config['train_dataset'])  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    valid_lmdb_path = config['lmdb_datasets'].get('valid_lmdb_path', None)
    validation_dataset = Dataset(valid_lmdb_path)
    # validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
        
    model = Model(**config['network_config']).to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler']['kwargs'])
    scheduler = WarmupLR(optimizer, **config['scheduler']['kwargs'])
    
    loss_func = Loss(**config['loss']).to(args.device)

    trainer = Trainer(config=config, model=model,optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)
        
        # save the config and codes
        if self.rank == 0:
            data = OmegaConf.create(config)
            OmegaConf.save(data, os.path.join(self.exp_path, 'config.yaml'))

            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(Path(__file__).parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        if self.resume:
            self._resume_checkpoint()

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-2]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self, epoch):
        total_loss = 0
        total_loss_s = 0
        total_loss_n = 0
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=125)

        for step, (noisy, clean) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            noise = noisy - clean

            enhanced, est_noise = self.model(noisy)
                
            loss_s = self.loss_func(enhanced, clean)
            loss_n = self.loss_func(est_noise, noise)
            if self.world_size > 1:
                loss_s = reduce_value(loss_s)
                loss_n = reduce_value(loss_n)
            total_loss_s += loss_s.item()
            total_loss_n += loss_n.item()
            loss = loss_s + loss_n
            total_loss += loss
            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.3f}, loss_s={:.3f}, loss_n={:.3f}'\
                                        .format(total_loss / step, total_loss_s / step, total_loss_n / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)
            self.writer.add_scalars('loss_s', {'loss_s': total_loss_s / step}, epoch)
            self.writer.add_scalars('loss_n', {'loss_n': total_loss_n / step}, epoch)


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_loss_s = 0
        total_loss_n = 0
        total_pesq_score = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=135)
        for step, (noisy, clean) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  
            noise = noisy - clean
            enhanced, est_noise = self.model(noisy)

            loss_s = self.loss_func(enhanced, clean)
            loss_n = self.loss_func(est_noise, noise)
            if self.world_size > 1:
                loss_s = reduce_value(loss_s)
                loss_n = reduce_value(loss_n)
            total_loss_s += loss_s.item()
            total_loss_n += loss_n.item()
            loss = loss_s + loss_n
            total_loss += loss.item()

            clean = clean.cpu().numpy()
            enhanced = enhanced.detach().cpu().numpy()
            noise = noise.detach().cpu().numpy()
            est_noise = est_noise.detach().cpu().numpy()
            clean_resample = librosa.resample(clean, orig_sr=self.config['samplerate'], target_sr=16000)
            enhanced_resample = librosa.resample(enhanced, orig_sr=self.config['samplerate'], target_sr=16000)
            def _safe_pesq(sr, clean, enhanced, mode):
                """安全的 PESQ 计算函数"""
                # 1. 检查音频有效性
                if np.max(np.abs(clean)) < 1e-5 or np.max(np.abs(enhanced)) < 1e-5:
                    print("One of the audio signals is too silent for PESQ calculation.")
                    print(f"Max clean: {np.max(np.abs(clean))}, Max enhanced: {np.max(np.abs(enhanced))}")
                    return np.nan
                return pesq(sr, clean, enhanced, mode)
            def _safe_normalize_audio(audio):
                peak = np.max(np.abs(audio))
                return audio / (peak + 1e-10) * 0.9  # 归一化到 90% 峰值  
            try:
                pesq_score_batch = Parallel(n_jobs=-1)(
                delayed(_safe_pesq)(16000, c, e, 'wb') \
                    for c, e in zip(_safe_normalize_audio(clean_resample), _safe_normalize_audio(enhanced_resample)))
            except Exception as e:
                print(f"PESQ 计算错误: {e}")
                continue

            pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            if self.world_size > 1:
                pesq_score = reduce_value(pesq_score)
            total_pesq_score += pesq_score
            
            if self.rank == 0 and (epoch==1 or epoch %10 == 0) and step <= 30:
                noisy_path = os.path.join(self.sample_path, 'sample_{}_noisy.wav'.format(step))
                clean_path = os.path.join(self.sample_path, 'sample_{}_clean.wav'.format(step))
                noise_path = os.path.join(self.sample_path, 'sample_{}_noise.wav'.format(step))
                enhanced_path = os.path.join(self.sample_path, 'sample_{}_enh_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                estnoise_path = os.path.join(self.sample_path, 'sample_{}_estn_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                if not os.path.exists(noisy_path):
                    noisy = noisy.cpu().numpy()
                    sf.write(noisy_path, noisy[0], samplerate=self.config['samplerate'])
                    sf.write(clean_path, clean[0], samplerate=self.config['samplerate'])
                    sf.write(noise_path, noise[0], samplerate=self.config['samplerate'])

                sf.write(enhanced_path, enhanced[0], samplerate=self.config['samplerate'])
                sf.write(estnoise_path, est_noise[0], samplerate=self.config['samplerate'])

            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'valid_loss={:.3f}, loss_s={:.3f}, loss_n={:.3f}, pesq={:.4f}'.format(
                total_loss / step, total_loss_s / step, total_loss_n / step, total_pesq_score / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars(
                'val_loss', {'val_loss': total_loss / step, 
                             'loss_s': total_loss_s / step,
                             'loss_n': total_loss_n / step,
                             'pesq': total_pesq_score / step}, epoch)

        return total_loss / step, total_pesq_score / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)
            
            if self.config['scheduler']['update_interval'] == 'epoch':
                if self.config['scheduler']['use_plateau']:
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

            print('------------Training for {} epochs is done!------------'.format(self.epochs))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = OmegaConf.load(args.config)
    config = OmegaConf.load('configs/cfg_train.yaml')
    # 将 OmegaConf 的 DictConfig/ListConfig 等转换为原生的 Python 容器（dict/list）
    # 这样 configs 中的 snd_db: [0, 15] 会成为 Python 列表 [0, 15]
    try:
        config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        # 若转换失败则保持原始 config（兼容性），后续可手动转换字段
        pass
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
