import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#==============================================

def dynamic_sym_std_quant(x, bit_num=4):
    # 定义量化的范围
    qmin = -2**(bit_num - 1)
    qmax = 2**(bit_num - 1) - 1

    std_scale = 2.6
    mean = x.mean()
    std = x.std()
    
    scale = torch.maximum((mean - std_scale * std).abs(),
                  (mean + std_scale * std).abs()) / (qmax - qmin + 1)
                  
    # 定义scale和zero_point，zero_point为0
    zero_point = 0.0

    # 量化步骤
    x_normalized = x / scale
    x_quantized = torch.round(x_normalized).clamp(qmin, qmax)

    return x_quantized, scale, zero_point
    

#先将int8反量化为浮点数
#再将浮点数量化为int4，返回量化结果和对应scale参数
def trans_int8toint4(x, scale, zp):
    x_fp32 = (x - zp) * scale
    return dynamic_sym_std_quant(x_fp32)

def my_conv(input, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if padding > 0:
        input = F.pad(input, (padding,padding,padding,padding))
    batch_size = input.shape[0]
    input_h, input_w = input.shape[2:4]
    kernel_h, kernel_w = kernel.shape[2:4]
    out_channel, in_channel = kernel.shape[0:2]
    output_h = math.floor((input_h - kernel_h) / stride + 1)
    output_w = math.floor((input_w - kernel_w) / stride + 1)

    unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride)
    input_vector = unfold(input)

    kernel_vector = kernel.reshape(kernel.shape[0], -1).T
    input_vector = input_vector.permute(0,2,1).contiguous()

    output = input_vector @ kernel_vector
    if bias != None:
        output = output + bias
    output = output.reshape(batch_size, output_h, output_w, out_channel).permute(0,3,1,2).contiguous()

    return output



#resnet50 第三层卷积数据
int8_1152 = np.load("1152.npy")
int8_1152 = torch.from_numpy(int8_1152)
int8_1152_scale = 0.0734466090798378
int8_1152_zp    = 0

int8_weight = np.load("layer1.0.conv2.weight.npy")
int8_weight = torch.from_numpy(int8_weight)
int8_weight_scale = 0.0086982985958457
int8_weight_zp = 0

int8_1159 = np.load("1159.npy")
int8_1159 = torch.from_numpy(int8_1159)
int8_1159_scale = 0.10853301733732224
int8_1159_zp = 0

int4_1152, int4_1152_scale, int4_1152_zp = trans_int8toint4(int8_1152, int8_1152_scale, int8_1152_zp)
int4_weight, int4_weight_scale, int4_weight_zp = trans_int8toint4(int8_weight, int8_weight_scale, int8_weight_zp)
int4_1159, int4_1159_scale, int4_1159_zp = trans_int8toint4(int8_1159, int8_1159_scale, int8_1159_zp)


np.save("int4_1152.npy", int4_1152.numpy())
print(f"int4_1152_scale:{int4_1152_scale}, int4_1152_zp:{int4_1152_zp}")
np.save("int4_1159.npy", int4_1159.numpy())
print(f"int4_1159_scale:{int4_1159_scale}, int4_1159_zp:{int4_1152_zp}")
np.save("int4_weight.npy", int4_weight.numpy())
print(f"int4_weight_scale:{int4_weight_scale}, int4_1152_zp:{int4_weight_zp}")

#fp32_1152 = (int8_1152 - int8_1152_zp) * int8_1152_scale
#fp32_weight = (int8_weight - int8_weight_zp) * int8_weight_scale
#my_conv(fp32_1152, fp32_weight, padding=1)




