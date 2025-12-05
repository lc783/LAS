import numpy as np
import torch
from torch import nn


class Invert2_10(nn.Module):
    def __init__(self):
        super(Invert2_10, self).__init__()
          
        sigmoid_h = np.array([-0.89521986, 1.8242345, 1.0888329, 1.2783841, 1.1793505, 5.402602,
                              3.556659, -0.45232677, 1.8296803, 2.5809617, 4.8217273, 2.7070105,
                              -0.76238555, 3.3387434, 1.7290733, 5.140637])
        sigmoid_d = np.array([-0.18134391, -0.07921064, -0.04980344, -0.02413579, 0.50927377, -0.0713518,
                              -0.64178044, -0.02694374, -0.06014722, -0.0580623, -1.1322998, -0.46733513,
                              -0.05247239, -0.04035916, -0.02469978, -0.01966147])
        sigmoid_T = np.array([2.821549, 2.1506927, 1.9763926, 2.5215197, -0.99999964, -1.4365319,
                              0.9995256, -1.347566, -2.1773431, -2.2306669, -1.4854108, 0.9888728,
                              -2.9239423, -3.8899376, -3.105946, -4.132591])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]    
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)    
            out = out + z * self.d[t]    

        out = out.view(or_shape) * signs
        return out


class Invert1_10(nn.Module):
    def __init__(self):
        super(Invert1_10, self).__init__()

        sigmoid_h = np.array([-0.6076018, 0.06285988, 0.26102483, -0.37457255, 0.3718868, 0.37248886,
                              0.63687307, -0.92578804, 1.9272704, 0.7057378, 0.33205885, 1.1465814,
                              -1.2022963, 1.5285196, 0.6882014, 1.6742821])
        sigmoid_d = np.array([0.38161004, -0.15619771, -0.01158753, 0.519088, -0.17966147, -0.05126573,
                              0.24982832, -0.3483371, -0.12764367, -0.05997599, -0.06374894, 0.08144259,
                              -0.05318592, -0.00351633, -0.03889612, -0.02578991])
        sigmoid_T = np.array([-0.09472599, 1.2887092, -0.9986847, -0.95269436, 1.1932085, 2.9062123,
                              -0.99365354, 1.9449068, 0.6202121, 0.41550368, 0.6942196, -1.0003381,
                              2.0298457, -0.99100643, 1.0268241, 0.6855806])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
              

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    
            out = torch.stack(out)
        return out

class Invert1_4(nn.Module):
    def __init__(self):
        super(Invert1_4, self).__init__()
  
        sigmoid_h = np.array([-0.38979334, 0.24928325, 0.73890483, 0.35679913, 0.21842594,-0.44996148,
  0.8016629 , 0.08536069, 0.42679605,-1.0037178 ,-0.12923315, 1.0277165 ,
  0.35668162, 0.2777678 , 0.31543776, 0.4160618 ])
        sigmoid_d = np.array([ 0.44319355, 0.5081881 ,-0.21380213,-0.11570027,-0.01263872,-0.09582443,
 -0.00509998,-0.05975275, 0.10377067, 0.06061111,-0.12617196,-0.0822822 ,
 -0.05644104,-0.03629385,-0.03424098,-0.01755796])
        sigmoid_T = np.array([-0.78549933,-0.94956565, 0.25978535, 0.14981902, 0.11088277, 0.4955158 ,
  1.0002508 , 0.8150474 ,-0.17535704,-0.99992806, 0.79916763, 0.26726273,
  0.16308218, 0.12552261, 0.17827137, 0.02839503])

          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
              

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    
            out = torch.stack(out)
        return out

class Invert4_10(nn.Module):
    def __init__(self):
        super(Invert4_10, self).__init__()
          
        sigmoid_h = np.array([-0.00181154, 0.8721661 , 0.9177631 , 0.9392744 , 0.5681609 , 0.9465831 ,
  0.6847087 , 0.45589155, 0.57916474, 0.7803396 , 0.28270212, 0.49239117,
  1.1224731 , 0.5738949 , 0.32048506, 0.2620882 ])
        sigmoid_d = np.array([ 0.0931013 , 0.09543603,-0.00957536,-0.02775419, 0.07635077,-0.02604962,
 -0.01608226,-0.0154707 ,-0.01741009,-0.00761568,-0.00868225,-0.01600825,
 -0.00795393,-0.0046836 ,-0.00339996,-0.00177163])
        sigmoid_T = np.array([-0.25367174,-0.35691947, 0.35702407, 1.8097845 ,-0.8933508 , 0.74517566,
  0.57702994, 0.56928945, 0.61470956, 0.43903926, 0.20668195, 0.6593264 ,
  0.35631987, 0.15981139,-0.12464668,-0.22194518])

        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
              

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    
            out = torch.stack(out)
        return out

invert1_10 = Invert1_10()
invert2_10 = Invert2_10()

invert1_4 = Invert1_4()
invert4_10 = Invert4_10()

def invert_tensor(x: torch.Tensor, fast=True) -> torch.Tensor:
      
      
    exponent = torch.floor(torch.log10(x))
    factor = torch.pow(10, exponent)
    scaled = x / factor

      
    reciprocal_scaled = invert1_10(scaled, fast)

      
    return reciprocal_scaled / factor


def invert_tensor2(x: torch.Tensor, fast=True) -> torch.Tensor:
      
      
    exponent = torch.floor(torch.log10(x))
    factor = torch.pow(10, exponent)
    scaled = x / factor

    hidden_states = scaled
    mask_1_4 = hidden_states<=4
    mask_4_10 = hidden_states>4
    hidden_states_1_4 = hidden_states[mask_1_4]
    hidden_states_4_10 = hidden_states[mask_4_10]

    hidden_states_1_4 = invert1_4(hidden_states_1_4,fast).to(x.dtype)
    hidden_states_4_10 = invert4_10(hidden_states_4_10,fast).to(x.dtype)

    result = torch.empty_like(x)
    K = hidden_states_1_4.shape[0]
    result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))
    result[:,mask_1_4] = hidden_states_1_4
    result[:,mask_4_10] = hidden_states_4_10

      
    return result / factor


class EXPPos(nn.Module):
    def __init__(self):
        super(EXPPos, self).__init__()
          

          
        sigmoid_h = np.array([-0.86567855, 0.21855816, 0.3086633 , 0.04801017, 0.35778293, 0.20322426,
  0.18277626, 0.09193517, 0.07528244, 0.068702  , 0.05387684, 0.05479147,
  1.3643376 ,-0.3182269 ,-2.7805114 ,-7.681703  ])
        sigmoid_d = np.array([0.5104444 ,0.62620413,0.1284339 ,0.56105524,0.29178998,0.24080864,
 0.12161193,0.07751098,0.08108865,0.04356307,0.07833251,0.09840941,
 0.06726651,0.06394249,0.3326615 ,0.64297193])
        sigmoid_T = np.array([ 0.8691826 , 0.7848423 , 0.57847524, 0.65401286, 0.3684752 , 0.29347295,
  0.1512695 , 0.07149795, 0.04385582, 0.00291296, 0.00250672, 0.9997504 ,
  0.02010613, 0.97479296,-0.28520948,-0.9983561 ])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out
class EXP3(nn.Module):
    def __init__(self):
        super(EXP3, self).__init__()
          
        sigmoid_h = np.array([-0.8399946, 0.99820644, 0.33414993, 0.19615063, 0.3125045, 0.11342224,
                              0.1539571, 0.21228108, 0.38308653, 0.4452268, 0.29108727, 0.41668856,
                              0.41155726, 0.22978419, 0.26166952, -0.48477188])
        sigmoid_d = np.array([1.1735343, 3.02147, 2.519381, 1.1005627, 2.1412673, 2.2726693,
                              2.0566351, 2.2742991, 2.040415, 1.2209179, 1.1530449, 0.91817755,
                              0.46067572, 0.4314381, 0.20695622, 0.1421418])
        sigmoid_T = np.array([-0.00161414, 1.1921781, 1.185069, 0.20014314, 0.96967727, 0.8780785,
                              0.7584549, 0.5879346, 0.35898587, 0.24437876, 0.1190711, -0.05131322,
                              -0.33453465, -0.54057884, -0.68599683, -0.1610864])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out


class EXP3_5(nn.Module):
    def __init__(self):
        super(EXP3_5, self).__init__()
          
        sigmoid_h = np.array([-0.95430815, -21.349306, -7.2316775, 1.2471976, 0.91875654,
                              1.2107203, 0.24415907, 0.16486089, 0.1135251, 0.27934995,
                              0.36055538, 7.578265, 0.13689142, 0.354173, 0.22085197,
                              0.86756164])
        sigmoid_d = np.array([29.153414, 14.275578, 5.994073, 6.4539547, 6.175543, 24.50532,
                              21.001366, 11.247643, 18.547016, 18.019056, 11.392004, 9.455236,
                              10.08966, 5.8203936, 3.552473, 2.294918])
        sigmoid_T = np.array([6.1954703, 5.2002587, -1.0000002, -1.0000017, -1.0000001, 1.2270416,
                              1.2228385, 1.0626469, 0.87558097, 0.60132205, 2.1559896, 0.36338305,
                              0.10540916, -0.09197062, -0.2888271, -1.1361874])
        
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape) * signs
        else:
            x_abs = x
            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out


class EXPneg10_0(nn.Module):
    def __init__(self):
        super(EXPneg10_0, self).__init__()
          
        sigmoid_h = np.array([-0.67344314, 0.4527931, 0.49219826, -1.4068147, 0.6613018, 1.3162614,
                              -0.92726743, 2.634861, 0.96277684, -0.38136783, 1.4028617, -0.3775043,
                              1.922271, -0.3901626, 0.64179343, -0.41200224])
        sigmoid_d = np.array([0.31594568, 0.24647555, -0.7487582, 0.24122885, 0.23569937, -0.01894833,
                              0.20568946, 0.06161513, -0.12247714, 0.07312343, -0.01228724, 0.05022796,
                              -0.05799114, 0.0241676, -0.01311035, 0.02739978])
        sigmoid_T = np.array([-0.3486977, -0.6059309, 0.9999999, -0.91124636, -1.2961993, 0.89930004,
                              -1.9198496, -2.7091455, -0.35776606, -2.9399333, 0.9082479, -3.3428566,
                              0.9956271, -3.5294924, 0.30579972, -3.9726608])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            x_abs = x
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape)
        else:
            x_abs = x

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out

exppos = EXPPos()
EXP3_5 = EXP3_5()
EXP3 = EXP3()
EXPneg10_0 = EXPneg10_0()


def get_exp(x, fast=True):
    if fast:
        hidden_states = torch.clamp(x, min=-10, max=5)
        mask_neg10_0 = (hidden_states >= -10) & (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0) & (hidden_states < 3)
        mask_3_5 = (hidden_states >= 3) & (hidden_states <= 5)
        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]
        hidden_states_3_5 = hidden_states[mask_3_5]

        hidden_states_neg10_0 = EXPneg10_0(hidden_states_neg10_0).to(x.dtype)
        hidden_states_0_3 = EXP3(hidden_states_0_3).to(x.dtype)
        hidden_states_3_5 = EXP3_5(hidden_states_3_5).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_neg10_0] = hidden_states_neg10_0
        result[mask_0_3] = hidden_states_0_3
        result[mask_3_5] = hidden_states_3_5
        return result
    else:
        hidden_states = torch.clamp(x, min=-10, max=5)
        mask_neg10_0 = (hidden_states >= -10) & (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0) & (hidden_states < 3)
        mask_3_5 = (hidden_states >= 3) & (hidden_states <= 5)
        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]
        hidden_states_3_5 = hidden_states[mask_3_5]

        hidden_states_neg10_0 = EXPneg10_0(hidden_states_neg10_0, fast=False).to(x.dtype)
        hidden_states_0_3 = EXP3(hidden_states_0_3, fast=False).to(x.dtype)
        hidden_states_3_5 = EXP3_5(hidden_states_3_5, fast=False).to(x.dtype)

        result = torch.empty_like(x)
        K = hidden_states_neg10_0.shape[0]
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

        result[:, mask_neg10_0] = hidden_states_neg10_0
        result[:, mask_0_3] = hidden_states_0_3
        result[:, mask_3_5] = hidden_states_3_5
        return result

  
def get_exp2(x, fast=True):
    if fast:
        hidden_states = torch.clamp(x, min=-10, max=5)
        mask_neg10_0 = (hidden_states >= -10) & (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0) & (hidden_states < 3)
        mask_3_5 = (hidden_states >= 3) & (hidden_states <= 5)
        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]
        hidden_states_3_5 = hidden_states[mask_3_5]

        hidden_states_neg10_0 = EXPneg10_0(hidden_states_neg10_0).to(x.dtype)
        hidden_states_0_3 = EXP3(hidden_states_0_3).to(x.dtype)
        hidden_states_3_5 = EXP3_5(hidden_states_3_5).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_neg10_0] = hidden_states_neg10_0
        result[mask_0_3] = hidden_states_0_3
        result[mask_3_5] = hidden_states_3_5
        return result
    else:
        hidden_states = torch.clamp(x, min=-80, max=80)
        mask_neg10_0 = (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0)

        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]


        neg_x = hidden_states_neg10_0
        abs_neg_x = -neg_x    
          
        n_neg = torch.floor(abs_neg_x)
        d_neg = abs_neg_x - n_neg
          
        exp_nd = (torch.tensor(2.71828, device=x.device) ** n_neg) * exppos(d_neg,fast=False)     
          

        neg = nolinearOperaterInv(exp_nd)

        n = torch.floor(hidden_states_0_3)    
        d = hidden_states_0_3 - n    
        exp_d = exppos(d,fast=False)    
        e = torch.tensor(2.71828, device=x.device)    
        exp_n = e ** n    
        hidden_states_0_3 = exp_n * exp_d

        result = torch.empty_like(x)
        K = neg.shape[0]
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

        result[:, mask_neg10_0] = neg
        result[:, mask_0_3] = hidden_states_0_3
        return result
    
def get_exp3(x, fast=True):
    if fast:
        hidden_states = torch.clamp(x, min=-10, max=5)
        mask_neg10_0 = (hidden_states >= -10) & (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0) & (hidden_states < 3)
        mask_3_5 = (hidden_states >= 3) & (hidden_states <= 5)
        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]
        hidden_states_3_5 = hidden_states[mask_3_5]

        hidden_states_neg10_0 = EXPneg10_0(hidden_states_neg10_0).to(x.dtype)
        hidden_states_0_3 = EXP3(hidden_states_0_3).to(x.dtype)
        hidden_states_3_5 = EXP3_5(hidden_states_3_5).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_neg10_0] = hidden_states_neg10_0
        result[mask_0_3] = hidden_states_0_3
        result[mask_3_5] = hidden_states_3_5
        return result
    else:
        hidden_states = torch.clamp(x, min=-80, max=80)
        mask_neg10_0 = (hidden_states < 0)
        mask_0_3 = (hidden_states >= 0)

        hidden_states_neg10_0 = hidden_states[mask_neg10_0]
        hidden_states_0_3 = hidden_states[mask_0_3]

        neg_x = hidden_states_neg10_0
        abs_neg_x = -neg_x    
          
        n_neg = torch.floor(abs_neg_x)
        d_neg = abs_neg_x - n_neg
          
        exp_nd = (torch.tensor(2.71828, device=x.device) ** n_neg) * exp0_1FS(d_neg,fast=False)     

          
        neg = nolinearOperaterInv(exp_nd)
        n = torch.floor(hidden_states_0_3)    
        d = hidden_states_0_3 - n    
        exp_d = exp0_1FS(d,fast=False)    
        e = torch.tensor(2.71828, device=x.device)    
        exp_n = e ** n    
        hidden_states_0_3 = exp_n * exp_d

        result = torch.empty_like(x)
        K = neg.shape[0]
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

        result[:, mask_neg10_0] = neg
        result[:, mask_0_3] = hidden_states_0_3
        return result

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        z = (v_scaled > 0).float()
        ctx.save_for_backward(v_scaled)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, = ctx.saved_tensors
        dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
        grad_input = grad_output * dz_dv_scaled
        return grad_input


  
class PsActivation(nn.Module):
    """
    Custom activation module that uses PS coding for spiking neural networks.
    """

    def __init__(self, dy=None, l=None, r=None, path=None):
        super().__init__()

        self.param_path = path
        self.hdT = torch.load(self.param_path)
          
        self.h = self.hdT["h"]
        self.d = self.hdT["d"]
        self.T = self.hdT["T"]
        self.b = self.hdT["b"]

          
        self.spike_count = 0
        self.neruon_count = 0

          
        dy = self.hdT["dy"]
        l = self.hdT["l"]
        r = self.hdT["r"]

    def reset_count(self):
        """
        Reset spike and neuron count to zero.
        """
        self.spike_count = 0
        self.neruon_count = 0

    def forward(self, x, fast=True):
        """
        Forward pass through the PS activation.
        """
        if self.h.device != x.device:
            self.h = self.h.to(x.device)
            self.d = self.d.to(x.device)
            self.T = self.T.to(x.device)
        sp = x.shape    
        x_flat = x.view(-1)    
        _h = self.h[:, 0]    
        
        idx = torch.searchsorted(_h, x_flat)
        idx = torch.clamp(idx, 1, len(_h) - 1)    
         
        left = _h[idx - 1]
        right = _h[idx]
    
        left_diff = torch.abs(x_flat - left)
        right_diff = torch.abs(x_flat - right)

        nearest = torch.where(left_diff < right_diff, left, right)
        nearest_idx = torch.where(left_diff < right_diff, idx - 1, idx)

        x = nearest.view(sp)
        idx = nearest_idx.view(sp)
        out, spikes = ps(x, self.h, self.d, self.T, self.b, idx, fast)

        return out


def ps(x, h, d, T, b, idx, fast=True):
    v = x.clone()    
    z = torch.zeros_like(x)    
    out = torch.zeros_like(x)    
    t = 1    
    K = len(d) - 1    
    spikes = 0    
    res = []

    while t <= K:
        z = torch.where(
            v - T[t] >= 0,    
            torch.ones_like(v),    
            torch.zeros_like(v),    
        )
        out += z * d[t]    
        if not fast:
            res.append(z * d[t])
        spikes += z.sum()    

        if t != K:
            v = h[idx, t + 1]

        t += 1    
    if not fast:
        res = torch.stack(res)
        mask = (res != 0)    
        nonzero_indices = torch.argmax(mask.int(), dim=0)    
        cols = torch.arange(res.size(-1))    

        res[nonzero_indices, cols] -= b
        return res, spikes
    out -= b    
    return out, spikes    

sqr0_00004 = PsActivation(
    path="./weight/sqr_1e-06_1e-05_0.004.pt")
sqr0_02 = PsActivation(
    path="./weight/sqr_8e-06_0_0.2.pt")
sqr02_1 = PsActivation(
    path="./weight/sqr_2e-05_0.2_1.pt")

sqr0_1 = PsActivation(
    path="./weight/sqr_3e-05_0_1.pt")
  
  
sqr0_3 = PsActivation(
    path="./weight/sqr_5e-05_1e-05_3.pt")
sqe0_10 = PsActivation(
    path="./weight/sqrinv_0.0001_0_10.pt")
sqe10_1500 = PsActivation(
    path="./weight/sqrinv_0.001_10_1500.pt")
sqe1500_7000 = PsActivation(
    path="./weight/sqrt_0.001_1500_7000.pt")
sqe100000 = PsActivation(
    path="./weight/sqr_0.005_7000_100000.pt")

gelu6_30 = PsActivation(
    path="./weight/gelu_0.001_-6_30.pt")


squre0_20 = PsActivation(
    path="./weight/squre_0.008_8_20.pt")

squre0_100 = PsActivation(
    path="./weight/sqrinv_0.3_0_100.pt")

squre100_1000 = PsActivation(
    path="./weight/sqrinv_20_0_1000.pt")
  
  
squre1000_3000 = PsActivation(
    path="./weight/sqrinv_50_1000_3000.pt")

exp0_1FS = PsActivation(
    path="./weight/sqrinv_5e-05_0_1.pt")
  
sliu10_30FS = PsActivation(
    path="./weight/sliu_0.0005_-10_30.pt")   
sliuNEG20_3FS = PsActivation(
    path="./weight/SELU_0.0001_-20_3.pt")   
sliu3_50FS = PsActivation(
    path="./weight/SELU_0.001_3_50.pt")   

  
def get_sqr(x, fast=True, K=None):    
    if fast:
        hidden_states = torch.clamp(x, min=0, max=1500)
        mask_0_10 = (hidden_states >= 0) & (hidden_states < 10)
        mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_10 = sqe0_10(hidden_states_0_10).to(x.dtype)
        hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_0_10] = hidden_states_0_10
        result[mask_10_1500] = hidden_states_10_1500
        return result
    else:
        hidden_states = torch.clamp(x, min=0.0001, max=7000)
        mask_0_10 = (hidden_states >= 0.0001) & (hidden_states < 10)
        mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
        mask_1500_7000 = (hidden_states > 1500) & (hidden_states <= 7000)

        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]
        hidden_states_1500_7000 = hidden_states[mask_1500_7000]

        hidden_states_0_10 = sqe0_10(hidden_states_0_10, fast=False).to(x.dtype)
        hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500, fast=False).to(x.dtype)
        hidden_states_1500_7000 = sqe1500_7000(hidden_states_1500_7000, fast=False).to(x.dtype)


        result = torch.empty_like(x)
        K = hidden_states_0_10.shape[0]
          
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))
          
        result[:, mask_0_10] = hidden_states_0_10
        result[:, mask_10_1500] = hidden_states_10_1500
        result[:, mask_1500_7000] = hidden_states_1500_7000

        return result


def get_sqr_acc(x, fast=True, K=None):    
    with torch.cuda.amp.autocast(enabled=False):
        if fast:
            hidden_states = torch.clamp(x, min=0.00001, max=7000)
            mask_0_1 = (hidden_states >= 0.00001) & (hidden_states < 1)
            mask_1_3 = (hidden_states >= 1) & (hidden_states <= 3)
              
            mask_0_10 =  (hidden_states > 3) & (hidden_states < 10)
            mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
            mask_1500_7000 = (hidden_states > 1500) & (hidden_states <= 7000)

            hidden_states_0_1 = hidden_states[mask_0_1]
            hidden_states_0_3 = hidden_states[mask_1_3]
            hidden_states_0_10 = hidden_states[mask_0_10]
            hidden_states_10_1500 = hidden_states[mask_10_1500]
            hidden_states_1500_7000 = hidden_states[mask_1500_7000] 

            hidden_states_0_1 = sqr0_1(hidden_states_0_1).to(x.dtype)

            hidden_states_0_3 = sqr0_3(hidden_states_0_3).to(x.dtype)
              
            hidden_states_0_10 = sqe0_10(hidden_states_0_10).to(x.dtype)  
             
            hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500).to(x.dtype)
            
            hidden_states_1500_7000 = sqe1500_7000(hidden_states_1500_7000).to(x.dtype)
              
            result = torch.empty_like(x)
            result[mask_0_1] = hidden_states_0_1
            result[mask_1_3] = hidden_states_0_3
            result[mask_0_10] = hidden_states_0_10
            result[mask_10_1500] = hidden_states_10_1500
            result[mask_1500_7000] = hidden_states_1500_7000
            return result
        else:
            hidden_states = torch.clamp(x, min=0.00001, max=7000)
            mask_0_1 = (hidden_states >= 0.00001) & (hidden_states <= 1)
            mask_1_3 = (hidden_states > 1) & (hidden_states <= 3)
              
            mask_0_10 =  (hidden_states < 10) & (hidden_states > 3)
            mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
            mask_1500_7000 = (hidden_states > 1500) & (hidden_states <= 7000)
            hidden_states_0_1 = hidden_states[mask_0_1]
            hidden_states_0_3 = hidden_states[mask_1_3]
            hidden_states_0_10 = hidden_states[mask_0_10]
            hidden_states_10_1500 = hidden_states[mask_10_1500]
            hidden_states_1500_7000 = hidden_states[mask_1500_7000] 

            hidden_states_0_1 = sqr0_1(hidden_states_0_1,fast=False).to(x.dtype)

            hidden_states_0_3 = sqr0_3(hidden_states_0_3,fast=False).to(x.dtype)
              
            hidden_states_0_10 = sqe0_10(hidden_states_0_10,fast=False).to(x.dtype)  
              
            hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500,fast=False).to(x.dtype)
              
            hidden_states_1500_7000 = sqe1500_7000(hidden_states_1500_7000,fast=False).to(x.dtype)

            result = torch.empty_like(x)
            K = hidden_states_0_10.shape[0]
              
            result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))
                           
            result[:,mask_0_1] = hidden_states_0_1
            result[:, mask_1_3] = hidden_states_0_3
            result[:, mask_0_10] = hidden_states_0_10
            result[:, mask_10_1500] = hidden_states_10_1500
            result[:, mask_1500_7000] = hidden_states_1500_7000
            return result
  
def get_sqr_acc2(x, fast=True, K=None):    
    with torch.cuda.amp.autocast(enabled=False):
        if fast:
            hidden_states = torch.clamp(x, min=0.00001, max=100000)
            mask_0_01 = hidden_states < 0.004
            mask_0_02 = (hidden_states >= 0.004) & (hidden_states < 0.2)
            mask_02_1 = (hidden_states >= 0.2) & (hidden_states < 1)
              
            mask_1_3 = (hidden_states >= 1) & (hidden_states <= 3)
              
            mask_0_10 =  (hidden_states > 3) & (hidden_states < 10)
            mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
            mask_1500_7000 = (hidden_states > 1500) & (hidden_states <= 7000)
            mask_100000 = (hidden_states <= 100000) & (hidden_states > 7000)

            hidden_states_0_01 = hidden_states[mask_0_01]
            hidden_states_0_02 = hidden_states[mask_0_02]
            hidden_states_02_1 = hidden_states[mask_02_1]
              
            hidden_states_0_3 = hidden_states[mask_1_3]
            hidden_states_0_10 = hidden_states[mask_0_10]
            hidden_states_10_1500 = hidden_states[mask_10_1500]
            hidden_states_1500_7000 = hidden_states[mask_1500_7000] 
            hidden_states_100000 = hidden_states[mask_100000]
             
            hidden_states_0_01 = sqr0_00004(hidden_states_0_01).to(x.dtype)   

            hidden_states_0_02 = sqr0_02(hidden_states_0_02).to(x.dtype)      
              
            hidden_states_02_1 = sqr02_1(hidden_states_02_1).to(x.dtype)
              
            hidden_states_0_3 = sqr0_3(hidden_states_0_3).to(x.dtype)
              
            hidden_states_0_10 = sqe0_10(hidden_states_0_10).to(x.dtype)  
              
            hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500).to(x.dtype)             

            hidden_states_1500_7000 = sqe1500_7000(hidden_states_1500_7000).to(x.dtype)             
            
            hidden_states_100000 = sqe100000(hidden_states_100000).to(x.dtype)             

            result = torch.empty_like(x)
            result[mask_0_01] = hidden_states_0_01
            result[mask_0_02] = hidden_states_0_02
            result[mask_02_1] = hidden_states_02_1
              
            result[mask_1_3] = hidden_states_0_3
            result[mask_0_10] = hidden_states_0_10
            result[mask_10_1500] = hidden_states_10_1500
            result[mask_1500_7000] = hidden_states_1500_7000
            result[mask_100000] = hidden_states_100000
            return result
        else:
            hidden_states = torch.clamp(x, min=0.00001, max=7000)
            mask_0_1 = (hidden_states >= 0.00001) & (hidden_states <= 1)
            mask_1_3 = (hidden_states > 1) & (hidden_states <= 3)
              
            mask_0_10 =  (hidden_states < 10) & (hidden_states > 3)
            mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
            mask_1500_7000 = (hidden_states > 1500) & (hidden_states <= 7000)
            hidden_states_0_1 = hidden_states[mask_0_1]
            hidden_states_0_3 = hidden_states[mask_1_3]
            hidden_states_0_10 = hidden_states[mask_0_10]
            hidden_states_10_1500 = hidden_states[mask_10_1500]
            hidden_states_1500_7000 = hidden_states[mask_1500_7000] 

            hidden_states_0_1 = sqr0_1(hidden_states_0_1,fast=False).to(x.dtype)

            hidden_states_0_3 = sqr0_3(hidden_states_0_3,fast=False).to(x.dtype)              

            hidden_states_0_10 = sqe0_10(hidden_states_0_10,fast=False).to(x.dtype)  
              
            hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500,fast=False).to(x.dtype)             

            hidden_states_1500_7000 = sqe1500_7000(hidden_states_1500_7000,fast=False).to(x.dtype)

            result = torch.empty_like(x)
            K = hidden_states_0_10.shape[0]
              
            result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))             
              
            result[:,mask_0_1] = hidden_states_0_1
            result[:, mask_1_3] = hidden_states_0_3
            result[:, mask_0_10] = hidden_states_0_10
            result[:, mask_10_1500] = hidden_states_10_1500
            result[:, mask_1500_7000] = hidden_states_1500_7000

            return result


  
def get_sqr2(x, fast=True, K=None):    
    if fast:
        hidden_states = torch.clamp(x, min=0, max=1500)
        mask_0_10 = (hidden_states >= 0) & (hidden_states < 10)
        mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_10 = sqe0_10(hidden_states_0_10).to(x.dtype)
        hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_0_10] = hidden_states_0_10
        result[mask_10_1500] = hidden_states_10_1500
        return result
    else:
        hidden_states = torch.clamp(x, min=0.0001, max=1500)
        mask_0_10 = (hidden_states >= 0.0001) & (hidden_states < 10)
        mask_10_1500 = (hidden_states >= 10) & (hidden_states <= 1500)
        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_10 = sqe0_10(hidden_states_0_10, fast=False).to(x.dtype)
        hidden_states_10_1500 = sqe10_1500(hidden_states_10_1500, fast=False).to(x.dtype)

        result = torch.empty_like(x)
        K = hidden_states_0_10.shape[0]
          
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))
                    
        result[:, mask_0_10] = hidden_states_0_10
        result[:, mask_10_1500] = hidden_states_10_1500
         
        return result

class spikeSqure0_3(nn.Module):
    def __init__(self):
        super(spikeSqure0_3, self).__init__()
          
        sigmoid_h = np.array([-0.6895311 , 0.12989718, 0.39969245, 0.09424607, 0.2640038 , 0.17132896,
  0.19190213, 0.11914219, 0.1977201 , 0.39886943, 0.35284662, 0.12098608,
  0.1462974 , 0.24224323, 0.20521896, 0.10770275])
        sigmoid_d = np.array( [0.48777387,0.01116391,0.09982327,0.43319267,0.58656216,0.60797155,
 0.36257643,0.5039462 ,0.8682856 ,0.7231767 ,0.22510543,0.28516194,
 0.22835907,0.14100431,0.05307972,0.04361422])
        sigmoid_T = np.array([ 1.8794003 ,-0.55938774, 0.19028088, 0.2471617 , 0.95013475, 0.84743863,
  0.70842886, 0.60115564, 0.49054465, 0.36998397, 0.23432286, 0.15082192,
  0.10303713, 0.06511012, 0.04114602,-0.11946911])         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            x_abs = x
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape)
        else:
            x_abs = x

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out

class Squre0_005(nn.Module):
    def __init__(self):
        super(Squre0_005, self).__init__()
          
        sigmoid_h = np.array( [-0.00443981, 0.0244156 ,-0.01410327, 0.01784681,-0.00239246, 0.06331808,
  0.01457577, 0.01753613, 0.01339214,-0.0131252 , 0.02026613, 0.00109124,
 -0.01996419, 0.00723148, 0.03052537,-0.00465259])
        sigmoid_d = np.array([-4.3056730e-02, 2.0701091e-05,-2.8386816e-02, 6.5610707e-02,
  7.1529418e-02,-1.0542608e-03, 1.2044241e-03, 7.7212957e-04,
 -7.2495628e-04, 1.0990113e-03, 8.7419833e-04,-3.1887756e-03,
 -1.9837018e-02, 3.0188679e-04, 2.2616754e-04,-1.7746047e-03])
        sigmoid_T = np.array([-0.1190194 , 0.03273299,-0.05586114, 0.07758415,-0.06072911, 0.04350465,
 -0.06501344,-0.07967568,-0.08437239,-0.07040057,-0.07470815, 0.01298669,
  0.12695177, 0.00617064,-0.08909994, 0.05016604])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape) * signs
        return out

class Squre005_01(nn.Module):
    def __init__(self):
        super(Squre005_01, self).__init__()

        sigmoid_h = np.array([-0.00301996, 0.01406131, 0.01828543, 0.01519527, 0.02173912, 0.02016546,
  0.00866903, 0.03278766, 0.03498204, 0.00712283, 0.00453039,-0.00784532,
  0.01212309, 0.00825764, 0.01493254, 0.01906155])
        sigmoid_d = np.array([ 0.00197194,-0.01671721, 0.00257154,-0.0372678 , 0.03564794, 0.00745469,
  0.01166203, 0.01102294,-0.05678081, 0.00933585, 0.00040685,-0.01868109,
  0.00044424, 0.04971262,-0.03885376, 0.00043014])
        sigmoid_T = np.array([ 0.06736092, 0.04292898, 0.05538262,-0.04594177,-0.02017328, 0.1154246 ,
  0.0062518 ,-0.00102335, 0.06101611,-0.01759881,-0.03017633, 0.01424571,
 -0.03390368,-0.0437814 ,-0.11353907,-0.04478091])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape) * signs
        return out

class Squre0_01(nn.Module):
    def __init__(self):
        super(Squre0_01, self).__init__()
        sigmoid_h = np.array([-0.00727064, 0.02973179, 0.07024261, 0.09523489, 0.05386556, 0.0581135 ,
  0.08542631, 0.05885677, 0.11285318, 0.03898872,-0.06981327, 0.1253458 ,
  0.15753898, 0.06028025,-0.07150588, 0.08730278])
        sigmoid_d = np.array([ 2.6471883e-03, 6.5944642e-03, 9.6436066e-04,-4.2568937e-02,
  1.2600903e-02, 1.0112234e-03,-9.2920298e-03, 9.4098366e-05,
 -4.5513660e-03,-2.1258840e-02, 4.6219504e-03, 4.5660823e-03,
 -1.0054999e-02,-2.3703028e-02, 2.7275824e-03, 1.6055545e-03])
        sigmoid_T = np.array([ 0.04104766, 0.03363906, 0.02748209, 0.09524764, 0.05120093, 0.06378917,
  0.08243488, 0.01002885,-0.03660668, 0.12500812,-0.03240077,-0.0475141 ,
 -0.02481392, 0.00858999,-0.05826636,-0.0609446 ])         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape) * signs
        return out

class Squre01_02(nn.Module):
    def __init__(self):
        super(Squre01_02, self).__init__()
          
        sigmoid_h = np.array([-0.00849385, 0.11065457,-0.02205543, 0.07415776, 0.1283021 , 0.11684933,
  0.12662175, 0.1362436 , 0.14139111, 0.14604805, 0.22716568, 0.15624896,
  0.00065499,-0.00175506,-0.04810934, 0.04944413])
        sigmoid_d = np.array([-0.04426918,-0.00312894, 0.05562779, 0.02765304, 0.02606837,-0.02705145,
 -0.00404149, 0.0316559 , 0.02473333, 0.0286518 , 0.02280941, 0.0027984 ,
  0.02111552, 0.00430653, 0.0048432 , 0.00589736])
        sigmoid_T = np.array([-0.17989106, 0.00238617,-0.01963474, 0.14340192, 0.19020334, 0.11338673,
  0.37603804, 0.03621124, 0.0430942 , 0.02228254, 0.00442842,-0.03809429,
  0.02185751,-0.02797154, 0.04041805,-0.05822638])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape) * signs
        return out

class Squre02_05(nn.Module):
    def __init__(self):
        super(Squre02_05, self).__init__()
          
        sigmoid_h = np.array( [-0.00285825, 0.09859432, 0.07954306, 0.08560576, 0.0910411 , 0.10557652,
  0.04608637, 0.07096123, 0.01989892, 0.03412642, 0.03321506, 0.01300904,
 -0.02348068, 0.06102338,-0.00509757, 0.00051896])
        sigmoid_d = np.array([ 0.04908372,-0.00948576, 0.00083943, 0.07961489, 0.08128168, 0.02395899,
  0.05197346, 0.01595683, 0.02185501, 0.01893722, 0.0074682 ,-0.00088913,
  0.01660369, 0.00298292, 0.0071363 , 0.00279426])
        sigmoid_T = np.array([ 0.20153396,-0.01304637,-0.19541621, 0.19903184, 0.1529713 ,-0.00479887,
  0.07102165, 0.02109929, 0.01368383,-0.00360851,-0.01454873,-0.03991705,
 -0.00440092,-0.06122432,-0.02989412,-0.04975629])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape) * signs
        return out

class spikeSqure0_1(nn.Module):   
    def __init__(self):
        super(spikeSqure0_1, self).__init__()
          
        sigmoid_h = np.array([-0.37860608, 1.463248  , 0.10793132, 0.37689486, 1.607159  ,-0.19645801,
  1.9294814 , 2.0232692 , 0.5594457 , 0.10542081, 0.71600085,-0.9493636 ,
 -0.9835639 , 0.01165338,-0.38191056,-1.4454263 ])
        sigmoid_d = np.array([ 0.21208926, 0.07186628,-0.02386915,-0.40844437, 0.02114153, 0.10371006,
 -0.02481472, 0.04288238,-0.05672836, 0.02667846, 0.17661859,-0.2705937 ,
 -0.19376335,-0.06404018,-0.03455378, 0.01753989])
        sigmoid_T = np.array([ 0.43828496, 0.3488404 ,-0.8391029 , 0.9999997 ,-0.1590641 , 0.10567223,
  1.0001101 ,-0.20675854, 0.48311934,-0.5016228 , 0.52276975, 0.99999756,
  0.99998015, 0.5117045 , 0.5433414 ,-0.99260676])         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            x_abs = x
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape)
        else:
            x_abs = x

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out

class spikeSqure2_8(nn.Module):
    def __init__(self):
        super(spikeSqure2_8, self).__init__()
          
        sigmoid_h = np.array( [-0.5268925 , 0.4195286 , 0.3772131 , 2.800345  , 0.6486724 , 0.7718605 ,
  0.52609473, 0.4238913 , 0.4271501 , 0.35683852, 0.33912426, 0.310338  ,
  0.39222103, 0.30863497, 0.45273697, 0.26908723])
        sigmoid_d = np.array([4.985456  ,3.940143  ,3.633856  ,5.1210046 ,4.055916  ,4.390043  ,
 5.612227  ,4.4306707 ,5.5045104 ,4.878589  ,4.2997766 ,4.73631   ,
 3.3738139 ,2.822326  ,1.6729311 ,0.81981134])
        sigmoid_T = np.array([ 5.9015956 , 5.127405  ,-1.000003  , 1.2043085 ,-0.12201342, 0.11561362,
  1.0937322 , 0.10791256, 1.0541725 , 0.76191485, 0.52203655, 0.31608614,
  0.04613298,-0.21407266,-0.54493636,-0.7457377 ])         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            x_abs = x
            x_abs = x_abs.view(-1)
            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]    

            out = out.view(or_shape)
        else:
            x_abs = x

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]    
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
                z = SpikeFunction.apply(v_scaled)    
                out.append(z * self.d[t])    

            out = torch.stack(out)
        return out

sque0_005 = Squre0_005()
sque0005_01 = Squre005_01()

sque0_01=Squre0_01()
sque01_02 = Squre01_02()
sque02_05 = Squre02_05()

getsqure0_1 = spikeSqure0_1()
getsqure = spikeSqure0_3()
getsqure2_8 = spikeSqure2_8()

def get_squre(x, fast=True, K=None):    
    x = x.abs()
    if fast:
        hidden_states = torch.clamp(x, min=0, max=3000)
        mask_0_1 = hidden_states<= 0.5
        mask_1_3 = (hidden_states>0.5) & (hidden_states <= 2)
        mask_3_10 =  (hidden_states > 2) &(hidden_states <= 8)

        mask_10_20 =  (hidden_states > 8) &(hidden_states <= 20)

        mask_3_100 =  (hidden_states > 20) &(hidden_states <= 100)
        mask_100_1000 = (hidden_states > 100) & (hidden_states <= 1000)
        mask_1000_3000 = (hidden_states > 1000) & (hidden_states <= 3000)
        hidden_states_0_1 = hidden_states[mask_0_1]
        hidden_states_1_3 = hidden_states[mask_1_3]
        hidden_states_3_10 = hidden_states[mask_3_10]

        hidden_states_10_20 = hidden_states[mask_10_20]

        hidden_states_3_100 = hidden_states[mask_3_100]
        hidden_states_100_1000 = hidden_states[mask_100_1000]
        hidden_states_1000_3000 = hidden_states[mask_1000_3000]

        hidden_states_0_1 = getsqure0_1(hidden_states_0_1).to(x.dtype)        
        hidden_states_1_3 = getsqure(hidden_states_1_3).to(x.dtype)       
        hidden_states_3_10 = getsqure2_8(hidden_states_3_10).to(x.dtype)        
          
        hidden_states_10_20  = squre0_20(hidden_states_10_20).to(x.dtype)         
        hidden_states_3_100 = squre0_100(hidden_states_3_100).to(x.dtype)
        hidden_states_100_1000 = squre100_1000(hidden_states_100_1000).to(x.dtype)
        hidden_states_1000_3000 = squre1000_3000(hidden_states_1000_3000).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_0_1] = hidden_states_0_1
        result[mask_1_3] = hidden_states_1_3
        result[mask_3_10] = hidden_states_3_10
        result[mask_10_20] = hidden_states_10_20

        result[mask_3_100] = hidden_states_3_100
        result[mask_100_1000] = hidden_states_100_1000
        result[mask_1000_3000] = hidden_states_1000_3000
        return result
    else:
        hidden_states = torch.clamp(x, min=0, max=1000)
        mask_0_10 = (hidden_states <= 100)
        mask_10_1500 = (hidden_states > 100) & (hidden_states <= 1000)
        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_10 = squre0_100(hidden_states_0_10, fast=False).to(x.dtype)
        hidden_states_10_1500 = squre100_1000(hidden_states_10_1500, fast=False).to(x.dtype)

        result = torch.empty_like(x)
        K = hidden_states_0_10.shape[0]
          
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))         
        result[:, mask_0_10] = hidden_states_0_10
        result[:, mask_10_1500] = hidden_states_10_1500       
        return result

def get_squre_post(x, fast=True, K=None):    
    x = x.abs()
    if fast:
        hidden_states = torch.clamp(x, min=0, max=7000)
        mask_0_01 = (hidden_states<= 0.1)
        mask_01_02 = (hidden_states>0.1) & (hidden_states <= 0.2)
        mask_02_05 = (hidden_states > 0.2) & (hidden_states <= 0.5)
          
        mask_1_3 = (hidden_states>0.5) & (hidden_states <= 2)
        mask_3_10 =  (hidden_states > 2) &(hidden_states <= 8)

        mask_10_20 =  (hidden_states > 8) &(hidden_states <= 20)

        mask_3_100 =  (hidden_states > 20) &(hidden_states <= 100)
        mask_100_1000 = (hidden_states > 100) & (hidden_states <= 1000)
        mask_1000_3000 = (hidden_states > 1000) & (hidden_states <= 3000)
        mask_3000_7000 = (hidden_states > 3000) & (hidden_states <= 7000)

        hidden_states_0_01 = hidden_states[mask_0_01]
        hidden_states_01_02 = hidden_states[mask_01_02]
        hidden_states_02_05 = hidden_states[mask_02_05]         
        hidden_states_1_3 = hidden_states[mask_1_3]
        hidden_states_3_10 = hidden_states[mask_3_10]

        hidden_states_10_20 = hidden_states[mask_10_20]

        hidden_states_3_100 = hidden_states[mask_3_100]
        hidden_states_100_1000 = hidden_states[mask_100_1000]
        hidden_states_1000_3000 = hidden_states[mask_1000_3000]
        hidden_states_3000_7000 = hidden_states[mask_3000_7000]
        
        hidden_states_0_01 = sque0_01(hidden_states_0_01).to(x.dtype)        
          
        hidden_states_01_02 = sque01_02(hidden_states_01_02).to(x.dtype)       
          
        hidden_states_02_05 = sque02_05(hidden_states_02_05).to(x.dtype)        
          
        hidden_states_1_3 = getsqure(hidden_states_1_3).to(x.dtype)       
          
        hidden_states_3_10 = getsqure2_8(hidden_states_3_10).to(x.dtype)                 

        hidden_states_10_20  = squre0_20(hidden_states_10_20).to(x.dtype)                   

        hidden_states_3_100 = squre0_100(hidden_states_3_100).to(x.dtype)               

        hidden_states_100_1000 = squre100_1000(hidden_states_100_1000).to(x.dtype)          

        hidden_states_1000_3000 = squre1000_3000(hidden_states_1000_3000).to(x.dtype)         

        hidden_states_3000_7000 = hidden_states_3000_7000**2

        result = torch.empty_like(x)
        result[mask_0_01] = hidden_states_0_01
        result[mask_01_02] = hidden_states_01_02
        result[mask_02_05] = hidden_states_02_05          
        result[mask_1_3] = hidden_states_1_3
        result[mask_3_10] = hidden_states_3_10
        result[mask_10_20] = hidden_states_10_20

        result[mask_3_100] = hidden_states_3_100
        result[mask_100_1000] = hidden_states_100_1000
        result[mask_1000_3000] = hidden_states_1000_3000
        result[mask_3000_7000] = hidden_states_3000_7000
        return result
    else:
        pass

  
def get_squre_post2(x, fast=True, K=None):    
    x = x.abs()
    if fast:
        hidden_states = torch.clamp(x, min=0, max=7000)
        mask_0_005 = (hidden_states<= 0.05)
        mask_0_01 = (hidden_states<= 0.1) & (hidden_states > 0.05)
        mask_01_02 = (hidden_states>0.1) & (hidden_states <= 0.2)
        mask_02_05 = (hidden_states > 0.2) & (hidden_states <= 0.5)
          
        mask_1_3 = (hidden_states>0.5) & (hidden_states <= 2)
        mask_3_10 =  (hidden_states > 2) &(hidden_states <= 8)

        mask_10_20 =  (hidden_states > 8) &(hidden_states <= 20)

        mask_3_100 =  (hidden_states > 20) &(hidden_states <= 100)
        mask_100_1000 = (hidden_states > 100) & (hidden_states <= 1000)
        mask_1000_3000 = (hidden_states > 1000) & (hidden_states <= 3000)
        mask_3000_7000 = (hidden_states > 3000) & (hidden_states <= 7000)

        hidden_states_0_005 = hidden_states[mask_0_005]
        hidden_states_0_01 = hidden_states[mask_0_01]
        hidden_states_01_02 = hidden_states[mask_01_02]
        hidden_states_02_05 = hidden_states[mask_02_05]
          
        hidden_states_1_3 = hidden_states[mask_1_3]
        hidden_states_3_10 = hidden_states[mask_3_10]

        hidden_states_10_20 = hidden_states[mask_10_20]

        hidden_states_3_100 = hidden_states[mask_3_100]
        hidden_states_100_1000 = hidden_states[mask_100_1000]
        hidden_states_1000_3000 = hidden_states[mask_1000_3000]
        hidden_states_3000_7000 = hidden_states[mask_3000_7000]

        hidden_states_0_005 = sque0_005(hidden_states_0_005).to(x.dtype)                  

        hidden_states_0_01 = sque0005_01(hidden_states_0_01).to(x.dtype)                            

        hidden_states_01_02 = sque01_02(hidden_states_01_02).to(x.dtype)       
          
        hidden_states_02_05 = sque02_05(hidden_states_02_05).to(x.dtype)        

        hidden_states_1_3 = getsqure(hidden_states_1_3).to(x.dtype)       

        hidden_states_3_10 = getsqure2_8(hidden_states_3_10).to(x.dtype)                 

        hidden_states_10_20  = squre0_20(hidden_states_10_20).to(x.dtype)                   

        hidden_states_3_100 = squre0_100(hidden_states_3_100).to(x.dtype)              

        hidden_states_100_1000 = squre100_1000(hidden_states_100_1000).to(x.dtype)          

        hidden_states_1000_3000 = squre1000_3000(hidden_states_1000_3000).to(x.dtype)          

        hidden_states_3000_7000 = hidden_states_3000_7000**2

        result = torch.empty_like(x)
        result[mask_0_005] = hidden_states_0_005
        result[mask_0_01] = hidden_states_0_01
        result[mask_01_02] = hidden_states_01_02
        result[mask_02_05] = hidden_states_02_05
          
        result[mask_1_3] = hidden_states_1_3
        result[mask_3_10] = hidden_states_3_10
        result[mask_10_20] = hidden_states_10_20

        result[mask_3_100] = hidden_states_3_100
        result[mask_100_1000] = hidden_states_100_1000
        result[mask_1000_3000] = hidden_states_1000_3000

        result[mask_3000_7000] = hidden_states_3000_7000

        return result
    else:
        pass

def get_squre2(x, fast=True, K=None):    
    x = x.abs()
    if fast:
        hidden_states = torch.clamp(x, min=0, max=1000)
        mask_0_3 = hidden_states<=3
        mask_3_10 =  (hidden_states > 3) &(hidden_states <= 100)
        mask_10_1500 = (hidden_states > 100) & (hidden_states <= 1000)
        hidden_states_0_3 = hidden_states[mask_0_3]
        hidden_states_3_10 = hidden_states[mask_3_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_3 = getsqure(hidden_states_0_3).to(x.dtype)
        hidden_states_3_10 = squre0_100(hidden_states_3_10).to(x.dtype)
        hidden_states_10_1500 = squre100_1000(hidden_states_10_1500).to(x.dtype)

        result = torch.empty_like(x)
        result[mask_0_3] = hidden_states_0_3
        result[mask_3_10] = hidden_states_3_10
        result[mask_10_1500] = hidden_states_10_1500
        return result
    else:
        hidden_states = torch.clamp(x, min=0, max=1000)
        mask_0_10 = (hidden_states <= 100)
        mask_10_1500 = (hidden_states > 100) & (hidden_states <= 1000)
        hidden_states_0_10 = hidden_states[mask_0_10]
        hidden_states_10_1500 = hidden_states[mask_10_1500]

        hidden_states_0_10 = squre0_100(hidden_states_0_10, fast=False).to(x.dtype)
        hidden_states_10_1500 = squre100_1000(hidden_states_10_1500, fast=False).to(x.dtype)

        result = torch.empty_like(x)
        K = hidden_states_0_10.shape[0]
          
        result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))         
          
        result[:, mask_0_10] = hidden_states_0_10
        result[:, mask_10_1500] = hidden_states_10_1500
         
        return result

class spikeGelu6_1(nn.Module):
    def __init__(self):
        super(spikeGelu6_1, self).__init__()
          
        sigmoid_h = np.array([-0.4542235 , 0.7169895 , 0.44056657, 0.24707365,-0.35661384, 2.5729966 ,
  4.1841526 , 1.098845  , 1.553965  , 1.2579314 , 1.1466882 , 1.9164954 ,
  2.775174  , 1.2413996 ,-3.303845  , 3.1743326 ])
        sigmoid_d = np.array([ 0.6011055 , 0.32070085, 0.1728974 , 0.07595864,-0.04305187,-0.0706069 ,
 -0.02941904,-0.13304795,-0.07925092,-0.07016345, 0.39347357, 0.43960708,
 -0.01215541, 0.28329173,-0.036314  ,-0.02618981])
        sigmoid_T = np.array([ 0.7052508 , 0.35964987, 0.10623224,-0.04634768, 0.8112458 ,-0.24774243,
 -0.44142142,-1.3476763 ,-0.4575439 ,-1.9938357 , 0.99998903, 0.99853265,
  0.82849   , 0.57890254,-2.021475  ,-2.7414792 ])         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        x_abs = x
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape)
        return out

class spikegelu0_30(nn.Module):
    def __init__(self):
        super(spikegelu0_30, self).__init__()
          
        sigmoid_h = np.array([-0.73437643, 3.319645  , 2.1290164 , 3.6901689 , 3.302721  , 3.2154958 ,
  2.9151256 , 3.1718695 , 3.1084518 , 2.6179547 , 2.235003  , 1.2864777 ,
  0.52327204,-0.08344932,-1.8249638 , 3.0859442 ])
        sigmoid_d = np.array( [ 3.3848417 ,-0.07072583, 3.691287  , 3.3074934 , 3.2120926 , 2.9196491 ,
  3.1727686 , 3.1054153 , 2.621943  , 2.2355895 , 1.2878408 , 0.52468026,
  0.12632068, 0.14482783, 0.3153498 , 0.13054803])
        sigmoid_T = np.array([ 3.1538513 ,-1.0211663 , 1.3714716 , 1.0718397 , 1.0049332 , 0.68078244,
  1.0151638 , 0.8858697 , 0.4037103 , 0.01490025,-0.90781677,-1.6859558 ,
 -1.9241625 ,-2.0441303 , 0.279356  , 0.1315174 ])
          
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        x_abs = x
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]                            
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)    
            z = SpikeFunction.apply(v_scaled)                
            out = out + z * self.d[t]                          

        out = out.view(or_shape)
        return out


gelu6_1 = spikeGelu6_1()

gelu0_30 = spikegelu0_30()

def get_gelu(x):
    hidden_states = torch.clamp(x,min=-6,max=30)
    mask_neg1= hidden_states<=1
    mask_1_30 = hidden_states>1
    hidden_states_neg1 = hidden_states[mask_neg1]
    hidden_states_1_30 = hidden_states[mask_1_30]

    hidden_states_neg1 = gelu6_1(hidden_states_neg1).to(x.dtype)
    hidden_states_1_30 = gelu0_30(hidden_states_1_30).to(x.dtype)

    result = torch.empty_like(x)
    result[mask_neg1] = hidden_states_neg1
    result[mask_1_30] = hidden_states_1_30
    return result

def get_gelu2(x):
    return gelu6_30(x)


def get_sliu(x):
    type = x.dtype
    x = x.to(torch.float32)
    x = torch.clamp(x,min=-10,max=30)
    return sliu10_30FS(x).to(type)

def get_sliu2(x):
    type = x.dtype
    x = x.to(torch.float32)
    hidden_states = torch.clamp(x,max=50)
    mask_neg1= (hidden_states<=3) & (hidden_states>-20)
    mask_1_30 = hidden_states>3
    hidden_states_neg1 = hidden_states[mask_neg1]
    hidden_states_1_30 = hidden_states[mask_1_30]

    hidden_states_neg1 = sliuNEG20_3FS(hidden_states_neg1).to(x.dtype)
    hidden_states_1_30 = sliu3_50FS(hidden_states_1_30).to(x.dtype)

    result = torch.zeros_like(x)
    result[mask_neg1] = hidden_states_neg1
    result[mask_1_30] = hidden_states_1_30    

    return result.to(type)


def nolinearOperaterInv(x):
      
    K = x.shape[0]
    res = []
    for i in range(K):
        if i == 0:
            Sl = x[i]
            mask = Sl > 0
            stata = Sl[mask]
            stata = 1 / stata
            O = torch.zeros_like(x[i])
            O[mask] = stata
        else:
            now = Sl + x[i]
            mask_now = now > 0
            now_stata = 1 / now[mask_now]    
            O_now = torch.zeros_like(x[i])
            O_now[mask_now] = now_stata

            last = Sl
            mask_last = last > 0
            last_stata = 1 / last[mask_last]    
            O_last = torch.zeros_like(x[i])
            O_last[mask_last] = last_stata

            O = O_now - O_last
            Sl = Sl + x[i]
        res.append(O)
    res = torch.stack(res)    
    return res


def SNNMatrixOperater(oA, oB):
    A = oA.to(torch.float32)
    B = oB.to(torch.float32)
    res = []
    with torch.cuda.amp.autocast(enabled=False):
      
        K = A.shape[0]

        sa = 0
        sb = 0
        for i in range(K):
            if i == 0:
                O = A[i] @ B[i]
                sa = A[i]
                sb = B[i]
            else:
                nsa = sa + A[i]
                nsb = sb + B[i]
                O = nsa @ nsb - sa @ sb
                sa = nsa
                sb = nsb
            res.append(O)
        res = torch.stack(res)    
    return res.to(oA.dtype)


def SNNMACOperater(oA, oB):
    A = oA.to(torch.float32)
    B = oB.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
      
        K = A.shape[0]
        res = 0
        sa = 0
        sb = 0
        for i in range(K):
            if i == 0:
                O = A[i] * B[i]
                sa = A[i]
                sb = B[i]
            else:
                nsa = sa + A[i]
                nsb = sb + B[i]
                O = nsa * nsb - sa * sb
                sa = nsa
                sb = nsb
            res += O
      
    return res.to(oA.dtype)

