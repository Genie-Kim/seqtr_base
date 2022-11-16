import torch
import torch.nn as nn
from seqtr.models import LAN_ENCODERS


@LAN_ENCODERS.register_module()
class LSTM(nn.Module):
    def __init__(self,
                 lstm_input_ch,
                 lstm_cfg=dict(type='gru',
                               num_layers=1,
                               dropout=0.,
                               hidden_size=512,
                               bias=True,
                               bidirectional=True,
                               batch_first=True),
                 output_cfg=dict(type="max")):
        super(LSTM, self).__init__()
        self.fp16_enabled = False

        assert lstm_cfg.pop('type') in ['gru']
        self.lstm = nn.GRU(
            **lstm_cfg, input_size=lstm_input_ch)

        output_type = output_cfg.pop('type')
        assert output_type in ['mean', 'default', 'max']
        self.output_type = output_type

    def forward(self, ref_expr):
        """Args:
            ref_expr_inds (tensor): [batch_size, max_token], 
                integer index of each word in the vocabulary,
                padded tokens are 0s at the last.

        Returns:
            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.
        """
        y_mask = torch.abs(ref_expr) == 0

        y_word = ref_expr

        y_word, h = self.lstm(y_word)

        if self.output_type == "mean":
            y = torch.cat(list(map(lambda feat, mask: torch.mean(
                feat[mask, :], dim=0, keepdim=True), y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "max":
            y = torch.cat(list(map(lambda feat, mask: torch.max(
                feat[mask, :], dim=0, keepdim=True)[0], y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "default":
            h = h.transpose(0, 1)
            y = h.flatten(1).unsqueeze(1)

        return y
