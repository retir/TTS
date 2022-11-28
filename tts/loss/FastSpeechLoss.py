import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, duration_predictor_loss
    
    def forward(self, batch, predicts):
        losses = {}
        losses["mel_loss"] = self.mse_loss(predicts["mel_spec"], batch["mel_target"])
        losses["duration_loss"] = self.l1_loss(predicts["duration"], batch["duration"].float())

        return losses
    

class PFastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target, pitch_target, predicted_pitch):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               torch.log1p(duration_predictor_target.float()))
        pitch_predictor_loss = self.mse_loss(torch.log1p(pitch_target), predicted_pitch)

        return mel_loss, duration_predictor_loss, pitch_predictor_loss
    
    def forward(self, batch, predicts):
        losses = {}
        losses["mel_loss"] = self.mse_loss(predicts["mel_spec"], batch["mel_target"])
        losses["duration_loss"] = self.l1_loss(predicts["duration"],  torch.log1p(batch["duration"].float()))
        losses["pitch_loss"] = self.mse_loss(predicts["pitches"],  torch.log1p(batch["pitches"]))

        return losses
    
class PEFastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target, pitch_target, predicted_pitch, energy_target, predicted_energy):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                               torch.log1p(duration_predictor_target.float()))
        pitch_predictor_loss = self.mse_loss(torch.log1p(pitch_target), predicted_pitch)
        energy_predictor_loss = self.mse_loss(torch.log1p(energy_target), predicted_energy)

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
    
    def forward(self, batch, predicts):
        losses = {}
        losses["mel_loss"] = self.l1_loss(predicts["mel_spec"], batch["mel_target"])
        losses["duration_loss"] = self.mse_loss(predicts["duration"],  torch.log1p(batch["duration"].float()))
        losses["pitch_loss"] = self.mse_loss(predicts["pitch"],  torch.log1p(batch["pitches"]))
        losses["energy_loss"] = self.mse_loss(predicts["energy"],  torch.log1p(batch["energy"]))

        return losses