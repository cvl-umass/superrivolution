# Adapted from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.out_fns import get_outfns


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, sr_type="model", new_out_size=None):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.outfns = get_outfns(tasks)
        self.sr_type = sr_type
        self.new_out_size = new_out_size
        # if self.sr_type == "input":
        #     self.upsample_inp = nn.Identity()
        if self.sr_type == "output":
            assert self.new_out_size is not None


    def forward(self, x, feat=False):
        out_size = x.size()[2:]
        if self.sr_type == "output":
            out_size = self.new_out_size
        # if self.sr_type == "input":
        #     x = self.upsample_inp(x)
        shared_representation = self.backbone(x)
        feats = shared_representation
        if isinstance(shared_representation, list):
            feats = shared_representation
            shared_representation = shared_representation[-1]
        decoded_out = {task: self.decoders[task](shared_representation) for task in self.tasks}
        # if self.sr_type == "output":
        #     decoded_out = {task: self.upsample_out(decoded_out[task]) for task in self.tasks}
        #     out_size = self.new_out_size
        
        if shared_representation.size()[2:] == out_size:
            out = {task: self.outfns[task](decoded_out[task]) for task in self.tasks}
            if feat:
                return out, feats
            return out
        else:
            out = {task: self.outfns[task](F.interpolate(decoded_out[task], out_size, mode='bilinear', align_corners=True)) for task in self.tasks}
            if feat:
                return out, feats
            return out
