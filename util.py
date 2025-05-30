import numpy as np
import torch
from torch import nn





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
            # x_abs = x_abs.view(-1)

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
            # x_abs = x_abs.view(-1)

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
        sigmoid_h = np.array([-0.00181154, 0.8721661, 0.9177631, 0.9392744, 0.5681609, 0.9465831,
                              0.6847087, 0.45589155, 0.57916474, 0.7803396, 0.28270212, 0.49239117,
                              1.1224731, 0.5738949, 0.32048506, 0.2620882])
        sigmoid_d = np.array([0.0931013, 0.09543603, -0.00957536, -0.02775419, 0.07635077, -0.02604962,
                              -0.01608226, -0.0154707, -0.01741009, -0.00761568, -0.00868225, -0.01600825,
                              -0.00795393, -0.0046836, -0.00339996, -0.00177163])
        sigmoid_T = np.array([-0.25367174, -0.35691947, 0.35702407, 1.8097845, -0.8933508, 0.74517566,
                              0.57702994, 0.56928945, 0.61470956, 0.43903926, 0.20668195, 0.6593264,
                              0.35631987, 0.15981139, -0.12464668, -0.22194518])


         
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
            # x_abs = x_abs.view(-1)

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

class Invert10_30(nn.Module):
    def __init__(self):
        super(Invert10_30, self).__init__()
        sigmoid_h = np.array([-0.00688555, 2.7319207 , 2.46037   , 1.2899711 , 0.90839106, 0.90053916,
  0.8651055 , 0.8316658 , 2.2194455 , 2.4012384 , 3.8117204 , 2.042905  ,
  4.7075944 , 4.8242817 , 3.6105917 , 3.0765817 ])
        sigmoid_d = np.array([ 0.03593914,-0.03824918,-0.00943666,-0.00837096, 0.01907302, 0.05347375,
  0.00287789,-0.01160881,-0.00965029,-0.01080756, 0.03090622,-0.00973814,
 -0.00739048,-0.00564446,-0.00379818,-0.00236025])
        sigmoid_T = np.array([-0.649072  , 5.653851  , 5.3024745 , 5.3065624 ,-0.98109895,-0.9999594 ,
 -0.99984664, 3.5049734 , 3.5992558 , 3.5661294 ,-1.0000166 , 1.0000184 ,
  0.44523025, 0.07517688,-0.44287837,-1.8136408 ])


         
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
            # x_abs = x_abs.view(-1)

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

class Invert30_60(nn.Module):
    def __init__(self):
        super(Invert30_60, self).__init__()
        sigmoid_h = np.array([-6.0949568e-03, 7.0675545e+00, 7.2954702e+00, 6.2444725e+00,
                              3.5912945e+00, 2.5436351e+00, 1.1306049e+00, 1.0479670e+00,
                              6.9566602e-01, 5.3732152e+00, 7.5771737e-01, 4.6541615e+00,
                              -6.2387747e-01, 4.9235511e+00, 4.3348312e+00, 5.0449753e-01])
        sigmoid_d = np.array([0.04131689, 0.00717232, -0.06977383, -0.06088164, -0.00226958, 0.02915924,
                              -0.0013556, 0.01610753, -0.00349431, 0.04134043, -0.00259233, 0.02245737,
                              -0.00228542, -0.0017479, -0.00110208, 0.00560706])
        sigmoid_T = np.array([-0.6098906, -0.50695634, 5.880563, 8.762766, 9.963368, -0.89375037,
                              6.5244784, -0.93218285, 9.19881, -0.9993793, 7.305162, -0.9999897,
                              7.655476, 6.5340242, 5.4307585, -0.9999983])

         
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
            # x_abs = x_abs.view(-1)

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

class Invert02_1(nn.Module):
    def __init__(self):
        super(Invert02_1, self).__init__()
        sigmoid_h = np.array([-0.72409785, 0.89465034, 1.6047162 ,-1.2706169 , 1.738143  , 0.01656771,
  0.13455446, 0.05566742, 0.16609071, 1.0180992 , 0.28325075, 0.52039206,
  2.6746373 , 7.898438  , 0.41684556,-2.0437558 ])
        sigmoid_d = np.array([ 6.8316736 , 1.7818565 , 5.590887  , 0.7893822 ,-0.60169536,-1.2686611 ,
 -0.10084324,-0.9199924 ,-1.1531742 ,-0.5291591 ,-0.24587825, 3.534518  ,
 -0.5946968 , 3.8478796 ,-5.33679   , 0.44648522])
        sigmoid_T = np.array([ 2.1057086 , 1.9633653 ,-1.000008  ,-0.9999368 ,-0.23426685,-0.19452742,
  0.5632431 ,-0.20608908,-0.27842647,-1.1037046 ,-1.2429477 , 1.0000042 ,
 -1.8416364 , 0.9763323 , 1.0000262 ,-1.4597445 ])


         
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
            # x_abs = x_abs.view(-1)

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

class Invert02_05(nn.Module):
    def __init__(self):
        super(Invert02_05, self).__init__()
        sigmoid_h = np.array([-0.8105155, -0.4380755, -0.7493194, -0.6881514, 0.05982972, -0.4554939,
                              0.01665272, -0.57653177, 0.66739935, 0.20097838, 0.08008745, -0.9236343,
                              1.1693529, 0.45021063, 0.04306528, 0.5974265])
        sigmoid_d = np.array([-0.14634968, -0.00227817, -0.1470271, -0.30400574, 0.18951349, -0.06620123,
                              1.3843102, 1.4218464, -0.0392062, -0.6671562, 1.012178, 0.9495134,
                              -0.61457163, -0.61489314, -0.41070494, -0.19526783])
        sigmoid_T = np.array([0.5462081, 0.56140864, 0.4561536, 0.41444692, 0.26295272, 0.7746396,
                              -1., -1., 1.2915751, 0.62044424, -0.9998573, -1.,
                              0.2440507, -0.09954549, -0.1127883, -0.12797347])
         
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
            # x_abs = x_abs.view(-1)

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

class Invert05_1(nn.Module):
    def __init__(self):
        super(Invert05_1, self).__init__()
        sigmoid_h = np.array([-0.34621114, 1.2097292 ,-0.03602187, 0.68631697, 0.46774933, 0.7782924 ,
  0.26072058, 0.75970656,-0.22251183,-2.9229019 , 0.17718989, 1.538257  ,
  0.25230032,-1.580791  ,-0.34686473,-1.0597632 ])
        sigmoid_d = np.array([ 0.07924628, 0.7255631 , 0.03879621, 0.03891919, 1.1721277 ,-0.508509  ,
  0.5537763 ,-0.02700002, 0.16735421,-0.22213821,-0.16337523,-0.0895365 ,
 -0.59149826,-0.07505188,-0.25674567, 0.39718035])
        sigmoid_T = np.array([ 2.284439  ,-0.9999996 , 2.300677  , 2.2847006 ,-0.9999999 ,-0.06219782,
  0.9988453 ,-0.20463064, 0.44629973, 0.08381487, 0.83336294,-0.02372687,
  0.04196908, 0.9560143 , 0.83289534, 0.989697  ])


         
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
            # x_abs = x_abs.view(-1)

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
invert10_30 = Invert10_30()
invert30_60 = Invert30_60()
invert02_1 = Invert02_1()
invert05_1 = Invert05_1()
invert02_05 = Invert02_05()


def invert_tensor_precise(x: torch.Tensor, fast=False) -> torch.Tensor:
    hidden_states = torch.clamp(x, min=0.2, max=60)
    mask_01_05 = (hidden_states <=0.5)
    mask_05_1 = (hidden_states <=1) & (hidden_states >0.5)
    mask_1_4 = (hidden_states<=4) & (hidden_states>1)
    mask_4_10 = (hidden_states > 4) & (hidden_states <= 10)
    mask_10_30 = (hidden_states > 10) & (hidden_states <= 30)
    mask_30_60 = (hidden_states > 30) & (hidden_states <= 60)

    hidden_states_02_05 = hidden_states[mask_01_05]
    hidden_states_05_1 = hidden_states[mask_05_1]
    hidden_states_1_4 = hidden_states[mask_1_4]
    hidden_states_4_10 = hidden_states[mask_4_10]
    hidden_states_10_30 = hidden_states[mask_10_30]
    hidden_states_30_60 = hidden_states[mask_30_60]

    hidden_states_02_05 = invert02_05(hidden_states_02_05, fast)
    hidden_states_02_1 = invert05_1(hidden_states_05_1, fast)
    hidden_states_1_4 = invert1_4(hidden_states_1_4, fast)
    hidden_states_4_10 = invert4_10(hidden_states_4_10, fast)
    hidden_states_10_30 = invert10_30(hidden_states_10_30, fast)
    hidden_states_30_60 = invert30_60(hidden_states_30_60, fast)

    result = torch.empty_like(x)
    K = hidden_states_1_4.shape[0]
    if K is None or K==0:
        K = hidden_states_02_1.shape[0]
    result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))
    result[:, mask_01_05] = hidden_states_02_05
    result[:, mask_05_1] = hidden_states_02_1
    result[:, mask_1_4] = hidden_states_1_4
    result[:, mask_4_10] = hidden_states_4_10
    result[:, mask_10_30] = hidden_states_10_30
    result[:, mask_30_60] = hidden_states_30_60

    return result


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

    def forward(self, x, fast=True):
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

        out, spikes = fs(x, self.h, self.d, self.T, self.b, idx, fast)

        return out


def fs(x, h, d, T, b, idx, fast=True):
    v = x.clone()  # Clone input tensor to maintain original values
    z = torch.zeros_like(x)  # Initialize tensor to track spikes
    out = torch.zeros_like(x)  # Initialize output tensor
    t = 1  # Initialize time step counter
    K = len(d) - 1  # Determine the number of steps
    spikes = 0  # Initialize spike counter
    res = []

    while t <= K:
        # Determine where input exceeds threshold and create spike indicators
        z = torch.where(
            v - T[t] >= 0,  # Compare input to threshold
            torch.ones_like(v),  # Set 1 where condition is met
            torch.zeros_like(v),  # Set 0 where condition is not met
        )
        out += z * d[t]  # Add step size to output where spikes occur
        if not fast:
            res.append(z * d[t])
        spikes += z.sum()  # Count total spikes

        if t != K:
            # Update input for next time step
            v = h[idx, t + 1]

        t += 1  # Move to next time step
    if not fast:
        res = torch.stack(res)
        mask = (res != 0)
        nonzero_indices = torch.argmax(mask.int(), dim=0)
        cols = torch.arange(res.size(-1))

        res[nonzero_indices, cols] -= b
        return res, spikes
    out -= b
    return out, spikes  # Return the output and the spike count


sqe0_10 = PsActivation(
    path="./weight/sqrinv_0.0001_0_10.pt")
sqe10_1500 = PsActivation(
    path="./weight/sqrinv_0.001_10_1500.pt")

gelu6_30 = PsActivation(
    path="./weight/gelu_0.001_-6_30.pt")

squre0_100 = PsActivation(
    path="./weight/sqrinv_0.3_0_100.pt")

squre100_1000 = PsActivation(
    path="./weight/sqrinv_20_0_1000.pt")

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
        sigmoid_h = np.array([-0.45029134, 0.13660802, 0.79116625, 0.35154283, 0.2300263 , 0.25131857,
  0.2859055 , 0.35303813, 0.09955073, 0.16905762, 0.43071112, 0.46997476,
  0.2730424 , 0.13731983, 0.30262393,-0.5251225 ])
        sigmoid_d = np.array([0.29946914,0.01354178,1.3328267 ,0.58273536,1.2930123 ,1.5332928 ,
 1.523752  ,0.44793314,0.5867976 ,0.9837378 ,0.8790002 ,0.45363167,
 0.22681575,0.2594645 ,0.07598496,0.12070131])
        sigmoid_T = np.array([ 1.1091447 ,-0.999358  , 0.9960277 , 0.35394624, 1.0220968 , 0.9840779 ,
  0.78471875, 0.52230215, 0.48891294, 0.41686916, 0.27483794, 0.04275339,
 -0.14358011,-0.23020536,-0.57749504, 0.10610177])
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

getsqure = spikeSqure0_3()

def get_squre(x, fast=True, K=None):
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
            now_stata = 1 / now[mask_now]  # function
            O_now = torch.zeros_like(x[i])
            O_now[mask_now] = now_stata

            last = Sl
            mask_last = last > 0
            last_stata = 1 / last[mask_last]  # function
            O_last = torch.zeros_like(x[i])
            O_last[mask_last] = last_stata

            O = O_now - O_last
            Sl = Sl + x[i]
        res.append(O)
    res = torch.stack(res)
    return res


if __name__ == "__main__":
    x_min, x_max = 0.5, 1
    batch_size = 100000
    x_np = np.linspace(x_min, x_max, batch_size).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)

    with torch.no_grad():
        y_approx = get_squre(x_tensor)
    import matplotlib.pyplot as plt
    y_true = x_tensor **2
    mse_loss = torch.mean((torch.abs(y_true - y_approx)) ** 2)
    plt.figure(figsize=(8, 4))
    plt.plot(x_np, y_true.numpy(), label='True Sigmoid')
    plt.plot(x_np, y_approx.numpy(), label='Approximation')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inference Test: True vs Approximated Sigmoid')
    plt.show()