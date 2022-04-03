# -- coding: utf-8 --
import pandas as pd
def sp_dis(hp):
    # sp shape [self.hp.site_num*self.hp.site_num, 15]
    sp = pd.read_csv(hp.file_sp,encoding='utf-8').values
    # dis shape [self.hp.site_num, self.hp.site_num]
    dis = pd.read_csv(hp.file_dis,encoding='utf-8').values
    return sp,dis