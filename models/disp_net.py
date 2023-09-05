import torch
import torch.nn as nn

from .disp_decoder import DispDecoder
from .disp_encoder import DispEncoder
from .pose_decoder import PoseDecoder
from .pose_encoder import PoseEncoder
from .utils import transformation_from_parameters
from .attention import PositionAttention, MultiHeadAttention
from .spm import SPM
from .depth_decoder import DepthDecoder

class DispNet(nn.Module):
    def __init__(self, opt):
        super(DispNet, self).__init__()

        self.opt = opt

        # networks
        self.DepthEncoder = DispEncoder(self.opt.depth_num_layers, pre_trained=True)
        self.DepthDecoder = DispDecoder(self.DepthEncoder.num_ch_enc)

        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers, True, num_input_images=2)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs['color_aug', 0, 0]))
        if self.training:
            outputs.update(self.predict_poses(inputs))
        return outputs

    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]
            pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
            axisangle, translation = self.PoseDecoder(pose_inputs)
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0],
                                                                            invert=(f_i < 0))
        return outputs

class DispAttetionNet(nn.Module):
    def __init__(self, opt):
        super(DispAttetionNet, self).__init__()

        self.opt = opt

        # networks
        self.DepthEncoder = DispEncoder(self.opt.depth_num_layers, pre_trained=True)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)

        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers, True, num_input_images=2)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)

        # self.PositionAttention = PositionAttention(self.DepthEncoder.num_ch_enc[-1])
        # self.MultiHeadAttention = MultiHeadAttention(self.DepthEncoder.num_ch_enc[-1], heads=4)
        # self.spm = SPM(self.DepthEncoder.num_ch_enc[-1])

    def forward(self, inputs):
        features = self.DepthEncoder(inputs['color_aug', 0, 0])
        # features[-1], attention_map = self.PositionAttention(features[-1])
        # features[-1], attention_map = self.MultiHeadAttention(features[-1])
        # features[-1], attention_map = self.spm(features[-1])
        outputs = self.DepthDecoder(features)

        # outputs['attention_map'] = attention_map
        if self.training:
            outputs.update(self.predict_poses(inputs))
        return outputs

    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]
            pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
            axisangle, translation = self.PoseDecoder(pose_inputs)
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0],
                                                                            invert=(f_i < 0))
        return outputs

