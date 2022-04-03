# -- coding: utf-8 --
import pandas as pd
import numpy as np
def in_out_deg(hp):
    # in_deg shape [self.hp.site_num*self.hp.site_num, 15]
    in_deg = pd.read_csv(hp.file_in_deg,encoding='utf-8').values[:,1]
    # out_deg shape [self.hp.site_num, self.hp.site_num]
    out_deg = pd.read_csv(hp.file_in_deg, encoding='utf-8').values[:,1]

    in_deg=np.reshape(in_deg,[1,-1])
    out_deg=np.reshape(out_deg,[1,-1])

    return in_deg,out_deg