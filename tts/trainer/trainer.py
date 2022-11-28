import os
import re
import torch
import waveglow
import text
import audio
import utils
import torchaudio
import numpy as np

from torch import nn
from tqdm import tqdm
from tts.utils import get_data, preprocess_english


class Trainer:
    def __init__(self, model, criterion, optimizer, logger, config, device, dataloader, len_epoch, lr_scheduler=None, skip_oom=True,):
        self.device = device
        self.config = config
        self.skip_oom = skip_oom
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.dataloader = dataloader
        self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        
        self._last_epoch = 0
        self.current_step = 0
        
        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        
        self.WaveGlow = utils.get_WaveGlow().to(self.device)
        self.raw_text = config['validation']['texts']
        if config["data"]["use_mfa"]:
            self.val_text = preprocess_english(self.raw_text, config["data"]["lexicon_path"], config['data']['text_cleaners'])
        else:      
            self.val_text = get_data(self.raw_text, config['data']['text_cleaners'])
        
        self.start_epoch = 1
        
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
       
    
    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "step" : self.current_step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
            
    
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.current_step = checkpoint["step"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            print(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer"] != self.config["optimizer"] or
                checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            print(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        print(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
    
    def synthesis(self, text, dur_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        text = np.array(text)
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().to(self.device)
        src_pos = torch.from_numpy(src_pos).long().to(self.device)

        with torch.no_grad():
            batch = {"text": sequence, "src_pos": src_pos}
            mel = self.model.forward(batch, dur_alpha=dur_alpha, pitch_alpha=pitch_alpha, energy_alpha=energy_alpha)
        return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
    
    def validate(self):
        print('Start validating')
        self.model.eval()
        self.logger.set_step(self.current_step, mode='test')
        
        save_dir = str(self.checkpoint_dir) + "/results_" + str(self.current_step).zfill(7)
        os.makedirs(save_dir, exist_ok=True)
        
        # classic
        for i, phn in tqdm(enumerate(self.val_text)):
            mel, mel_cuda = self.synthesis(phn)

            waveglow.inference.inference(
                mel_cuda, self.WaveGlow,
                f"{save_dir}/{self.raw_text[i][:5]}_classic.wav"
            )

            audio_tens, sr = torchaudio.load(f"{save_dir}/{self.raw_text[i][:5]}_classic.wav")
            self.logger.add_audio(f'{self.raw_text[i][:5]}_classic', audio_tens, sr)
        
        
        #val duration
        for dur in self.config['validation']['durations']:
            for i, phn in tqdm(enumerate(self.val_text)):
                mel, mel_cuda = self.synthesis(phn, dur_alpha=dur)

                waveglow.inference.inference(
                    mel_cuda, self.WaveGlow,
                    f"{save_dir}/{self.raw_text[i][:5]}_d={dur}.wav"
                )
                
                audio_tens, sr = torchaudio.load(f"{save_dir}/{self.raw_text[i][:5]}_d={dur}.wav")
                self.logger.add_audio(f'{self.raw_text[i][:5]}_d={dur}', audio_tens, sr)
         
        #val pitch
        for pitch_alpha in self.config['validation']['pitch']:
            for i, phn in tqdm(enumerate(self.val_text)):
                mel, mel_cuda = self.synthesis(phn, pitch_alpha=pitch_alpha)

                waveglow.inference.inference(
                    mel_cuda, self.WaveGlow,
                    f"{save_dir}/{self.raw_text[i][:5]}_p={pitch_alpha}.wav"
                )
                
                audio_tens, sr = torchaudio.load(f"{save_dir}/{self.raw_text[i][:5]}_p={pitch_alpha}.wav")
                self.logger.add_audio(f'{self.raw_text[i][:5]}_p={pitch_alpha}', audio_tens, sr)
        
        #vall energy
        for energy_alpha in self.config['validation']['energy']:
            for i, phn in tqdm(enumerate(self.val_text)):
                mel, mel_cuda = self.synthesis(phn, energy_alpha=energy_alpha)

                waveglow.inference.inference(
                    mel_cuda, self.WaveGlow,
                    f"{save_dir}/{self.raw_text[i][:5]}_e={energy_alpha}.wav"
                )
                
                audio_tens, sr = torchaudio.load(f"{save_dir}/{self.raw_text[i][:5]}_e={energy_alpha}.wav")
                self.logger.add_audio(f'{self.raw_text[i][:5]}_e={energy_alpha}', audio_tens, sr)
        
        #val all
        zipped = zip(self.config['validation']['durations'], self.config['validation']['pitch'], self.config['validation']['energy'])
        for dur_alpha, pitch_alpha, energy_alpha in zipped:
            for i, phn in tqdm(enumerate(self.val_text)):
                mel, mel_cuda = self.synthesis(phn, energy_alpha=energy_alpha, pitch_alpha=pitch_alpha, dur_alpha=dur_alpha)

                waveglow.inference.inference(
                    mel_cuda, self.WaveGlow,
                    f"{save_dir}/{self.raw_text[i][:5]}_all={energy_alpha}.wav"
                )
                
                audio_tens, sr = torchaudio.load(f"{save_dir}/{self.raw_text[i][:5]}_all={energy_alpha}.wav")
                self.logger.add_audio(f'{self.raw_text[i][:5]}_all={energy_alpha}', audio_tens, sr)
                
        print('End validating')
        self.model.train()
        self.logger.set_step(self.current_step)
    
    
    def prepare_batch(self, batch):
        for k, v in batch.items():
            if type(v) == torch.Tensor:
                if k in ["mel_target", "pitches", "energy"]:
                    batch[k] = v.float().to(self.device)
                else:
                    batch[k] = v.long().to(self.device)
        return batch
        
    
    def train(self):
        tqdm_bar = tqdm(total=self.config['trainer']['epochs'] * len(self.dataloader) * self.config['data']['batch_expand_size'] - self.current_step)
        self.model.train()
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            for i, batchs in enumerate(self.dataloader):
                for j, db in enumerate(batchs):
                    self.current_step += 1
                    tqdm_bar.update(1)

                    self.logger.set_step(self.current_step)

                    # Get Data
#                     character = db["text"].long().to(self.device)
#                     mel_target = db["mel_target"].float().to(self.device)
#                     duration = db["duration"].int().to(self.device)
#                     mel_pos = db["mel_pos"].long().to(self.device)
#                     src_pos = db["src_pos"].long().to(self.device)
#                     pitch_target = db["pitches"].float().to(self.device)
#                     energy_target = db["energy"].float().to(self.device)
#                     max_mel_len = db["mel_max_len"]
                    
                    batch = self.prepare_batch(db)
                    
                    # Forward
#                     mel_output, duration_predictor_output, predicted_pitch, predicted_energy = self.model(
#                                                                   character,
#                                                                   src_pos,
#                                                                   mel_pos=mel_pos,
#                                                                   mel_max_length=max_mel_len,
#                                                                   length_target=duration,
#                                                                   true_pitch=pitch_target,
#                                                                   true_energy=energy_target)

#                     # Calc Loss
#                     mel_loss, duration_loss, pitch_loss, energy_loss = self.criterion(mel_output,
#                                                             duration_predictor_output,
#                                                             mel_target,
#                                                             duration,
#                                                             pitch_target,
#                                                             predicted_pitch,
#                                                             energy_target,
#                                                             predicted_energy)
                    outputs = self.model(batch)
                    losses = self.criterion(batch, outputs)
                    total_loss = sum(losses.values())
                    for loss_name, loss_val in losses.items():
                        loss_numpy = loss_val.detach().cpu().numpy()
                        self.logger.add_scalar(loss_name, loss_numpy)
#                     total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                    # Logger
#                     t_l = total_loss.detach().cpu().numpy()
#                     m_l = mel_loss.detach().cpu().numpy()
#                     d_l = duration_loss.detach().cpu().numpy()
#                     p_l = pitch_loss.detach().cpu().numpy()
#                     e_l = energy_loss.detach().cpu().numpy()

#                     self.logger.add_scalar("duration_loss", d_l)
#                     self.logger.add_scalar("mel_loss", m_l)
#                     self.logger.add_scalar("total_loss", t_l)
#                     self.logger.add_scalar("pitch_loss", p_l)
#                     self.logger.add_scalar("energy_loss", e_l)

                    # Backward
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['trainer']['grad_norm_clip'])

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    last_lr = self.lr_scheduler.get_last_lr()[-1]
                    self.logger.add_scalar("lr", last_lr)
                    
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
            if epoch % self.config['validation']['val_step'] == 0:
                self.validate()