import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from tts.model.attention import MultiHeadAttention, PositionwiseFeedForward, PLNMultiHeadAttention
from tts.utils.model_utils import create_alignment, Transpose, get_non_pad_mask, get_attn_key_pad_mask, get_mask_from_lengths


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
    

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
    
    
class FFTBlockPLN(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(FFTBlockPLN, self).__init__()
        self.slf_attn = PLNMultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_k * n_head)
        self.ln2 = nn.LayerNorm(d_k * n_head)
        #self.ln1 = ScaleNorm(d_model**0.5)
        #self.ln2 = ScaleNorm(d_model**0.5)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        skip = enc_input
        
        x = self.ln1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            x, x, x, mask=slf_attn_mask)
        
        skip2 = skip + enc_output
        
        x = self.ln2(skip2)
        
        if non_pad_mask is not None:
            x *= non_pad_mask

        enc_output = self.pos_ffn(x)
        enc_output = enc_output + skip2
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
    

class Variance_adaptor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(Variance_adaptor, self).__init__()

        self.input_size = model_config['encoder_dim']
        self.filter_size = model_config['duration_predictor_filter_size']
        self.kernel = model_config['duration_predictor_kernel_size']
        self.conv_output_size = model_config['duration_predictor_filter_size']
        self.dropout = model_config['duration_predictor_dropout']

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
    
    
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = Variance_adaptor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (torch.expm1(duration_predictor_output) * alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.Tensor([i+1 for i in range(output.size(1))])]).long().to('cuda:0')
            return output, mel_pos
    
    
class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        len_max_seq=model_config["max_seq_len"]
        n_position = len_max_seq + 1
        n_layers = model_config["encoder_n_layer"]
        self.PAD = model_config["PAD"]

        self.src_word_emb = nn.Embedding(
            model_config["vocab_size"],
            model_config["encoder_dim"],
            padding_idx=model_config["PAD"]
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config["encoder_dim"],
            padding_idx=model_config["PAD"]
        )

        self.layer_stack = nn.ModuleList([FFTBlockPLN(
            model_config['encoder_dim'],
            model_config['encoder_conv1d_filter_size'],
            model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['fft_conv1d_kernel'],
            model_config['fft_conv1d_padding'],
            dropout=model_config["dropout"]
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, PAD=self.PAD)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq=model_config["max_seq_len"]
        n_position = len_max_seq + 1
        n_layers = model_config["decoder_n_layer"]
        self.PAD = model_config["PAD"]

        self.position_enc = nn.Embedding(
            n_position,
            model_config["decoder_dim"],
            padding_idx=model_config["PAD"],
        )

        self.layer_stack = nn.ModuleList([FFTBlockPLN(
            model_config["decoder_dim"],
            model_config["decoder_conv1d_filter_size"],
            model_config["decoder_head"],
            model_config["decoder_dim"] // model_config["decoder_head"],
            model_config["decoder_dim"] // model_config["decoder_head"],
            model_config['fft_conv1d_kernel'],
            model_config['fft_conv1d_padding'],
            dropout=model_config["dropout"]
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, PAD=self.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
    

class FastSpeech1(nn.Module):
    """ FastSpeech """

    def __init__(self, **model_config):
        super(FastSpeech1, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['decoder_dim'], model_config['num_mels'])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward2(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):

        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output
        else:
            output, mel_pos = self.length_regulator(x,alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            
            return output
    def forward(self, batch, alpha=1.0):

        x, non_pad_mask = self.encoder(batch["text"], batch["src_pos"])
        if self.training:
            outputs = {}
            output, outputs["duration"] = self.length_regulator(x, alpha, batch['duration'], batch["mel_max_len"])
            output = self.decoder(output, batch['mel_pos'])
            output = self.mask_tensor(output, batch['mel_pos'], batch["mel_max_len"])
            outputs["mel_spec"] = self.mel_linear(output)
            return outputs
        else:
            output, mel_pos = self.length_regulator(x,alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            
            return output
        
        
class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, **model_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)
        
        # pitch
        self.pitch_predictor = Variance_adaptor(model_config)
        self.pitch_embedding = nn.Embedding(model_config['pitch_bins'], model_config["encoder_dim"])
        self.pitch_bins = nn.Parameter(
            torch.linspace(model_config['pitch_min'], model_config['pitch_max'], model_config['pitch_bins'] - 1),
        requires_grad=False) # to be on the same device as model
        self.pitch_mean = model_config['pitch_mean']
        self.pitch_std = model_config['pitch_std']
        
        # energy
        self.energy_predictor = Variance_adaptor(model_config)
        self.energy_embedding = nn.Embedding(model_config['energy_bins'], model_config["encoder_dim"])
        self.energy_bins = nn.Parameter(
            torch.linspace(model_config['energy_min'], model_config['energy_max'], model_config['energy_bins'] - 1),
        requires_grad=False) 

        self.mel_linear = nn.Linear(model_config['decoder_dim'], model_config['num_mels'])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward2(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, true_pitch=None, true_energy=None, dur_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):

        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            
            output, duration_predictor_output = self.length_regulator(x, dur_alpha, length_target, mel_max_length)
            out_mask = get_non_pad_mask(mel_pos, PAD=self.decoder.PAD)
            
            # predict pitch
            predicted_pitch = self.pitch_predictor(output) 
            predicted_pitch = out_mask[:, :, 0] * predicted_pitch
            pitch_embed = self.pitch_embedding(torch.bucketize(true_pitch, self.pitch_bins))
            output = output + pitch_embed
            
            # predict energy
            predicted_energy = self.energy_predictor(output)
            predicted_energy = out_mask[:, :, 0] * predicted_energy
            energy_embed = self.energy_embedding(torch.bucketize(true_energy, self.energy_bins))
            output = output + energy_embed
            
            #decoder
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            
            return output, duration_predictor_output, predicted_pitch, predicted_energy
        else:
            output, mel_pos = self.length_regulator(x, dur_alpha)
            out_mask = get_non_pad_mask(mel_pos, PAD=self.decoder.PAD)
            
            # predict pitch
            predicted_pitch = torch.expm1(self.pitch_predictor(output)) * pitch_alpha 
            predicted_pitch = out_mask[:, :, 0] * predicted_pitch
            pitch_embed = self.pitch_embedding(torch.bucketize(predicted_pitch, self.pitch_bins))
            output = output + pitch_embed
            #output = output * out_mask
            
            # predict energy
            predicted_energy = torch.expm1(self.energy_predictor(output)) * energy_alpha 
            predicted_energy = out_mask[:, :, 0] * predicted_energy
            energy_embed = self.energy_embedding(torch.bucketize(predicted_energy, self.energy_bins))
            output = output + energy_embed
            
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            
            return output
        
    def forward(self, batch, dur_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        outputs = {}
        x, non_pad_mask = self.encoder(batch["text"], batch["src_pos"])
        
        if self.training:      
            output, outputs['duration'] = self.length_regulator(x, dur_alpha, batch['duration'], batch["mel_max_len"])
            out_mask = get_non_pad_mask(batch["mel_pos"], PAD=self.decoder.PAD)
            
            # predict pitch
            predicted_pitch = self.pitch_predictor(output) 
            outputs["pitch"] = out_mask[:, :, 0] * predicted_pitch
            pitch_embed = self.pitch_embedding(torch.bucketize(batch["pitches"], self.pitch_bins))
            output = output + pitch_embed
            
            # predict energy
            predicted_energy = self.energy_predictor(output)
            outputs["energy"] = out_mask[:, :, 0] * predicted_energy
            energy_embed = self.energy_embedding(torch.bucketize(batch["energy"], self.energy_bins))
            output = output + energy_embed
            
            #decoder
            output = self.decoder(output, batch["mel_pos"])
            output = self.mask_tensor(output, batch["mel_pos"], batch["mel_max_len"])
            outputs["mel_spec"] = self.mel_linear(output)
            
            return outputs
        else:
            output, mel_pos = self.length_regulator(x, dur_alpha)
            out_mask = get_non_pad_mask(mel_pos, PAD=self.decoder.PAD)
            
            # predict pitch
            predicted_pitch = torch.expm1(self.pitch_predictor(output)) * pitch_alpha 
            predicted_pitch = out_mask[:, :, 0] * predicted_pitch
            pitch_embed = self.pitch_embedding(torch.bucketize(predicted_pitch, self.pitch_bins))
            output = output + pitch_embed
            #output = output * out_mask
            
            # predict energy
            predicted_energy = torch.expm1(self.energy_predictor(output)) * energy_alpha 
            predicted_energy = out_mask[:, :, 0] * predicted_energy
            energy_embed = self.energy_embedding(torch.bucketize(predicted_energy, self.energy_bins))
            output = output + energy_embed
            
            # decoder
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            
            return output
