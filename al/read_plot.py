import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import loadtxt
import jsonlines as js

def read_log(beyond_mlp_cifar):
    beyond_mlp_cifar_label = [] 
    beyond_mlp_cifar_acc = [] 
    with open(beyond_mlp_cifar, "r") as f: 
        lines = f.readlines() 
        for line in lines: 
            if "testing accuracy" in line: 
                acc_label = line.split() 
                beyond_mlp_cifar_label.append(int(acc_label[0])) 
                beyond_mlp_cifar_acc.append(float(acc_label[3]))
    return beyond_mlp_cifar_label, beyond_mlp_cifar_acc

def read_rc(beyond_mlp_cifar):
    label = []
    em = [] 
    f1 = [] 
    
    with js.open(beyond_mlp_cifar, "r") as f: 
        for line in f: 
                label.append(line['round'] )
                em.append(line['exact']) 
                f1.append(line['f1'])
    return label, em, f1

def read_rc_log(beyond_mlp_cifar):
    if not os.path.isfile(beyond_mlp_cifar):
        print("no file")
        return
    print(beyond_mlp_cifar)
    rd = []
    em=[]
    fs=[]
    with open(f1, "r") as f: 
        lines = f.readlines() 
        for line in lines: 
            if "exact" in line: 
                tmp = line.split() 
                
                rd.append(int(tmp[1]))
                em.append(float(tmp[3])) 
                fs.append(float(tmp[5]))
                print('rd {} em {:2f} fs {:2f}'.format(int(tmp[1]), float(tmp[3]), float(tmp[5])))
    return rd, em, fs

# example
ipath = '../'
margin_r,margin_em, margin_f=read_rc(os.path.join(ipath, 'margin_i_40_2_epoch.jsonl'))
random_r,random_em, random_f=read_rc(os.path.join(ipath, 'random_i_40_2_epoch.jsonl'))
least_r, least_em, least_f=read_rc(os.path.join(ipath, 'least_i_40_2_epoch.jsonl'))
entro_r,entro_em, entro_f=read_rc(os.path.join(ipath, 'entropy_i_40_2_epoch.jsonl'))
bald_r,bald_em,bald_f=read_rc(os.path.join(ipath, 'bald_dropout_i_40_2_epoch.jsonl'))
# n = 12
n = 41
emf = {'#Labels queried':(np.array(margin_r)*2000+1000)[:n],'Badge': bald_f[:n], 'Conf': least_f[:n], \
      'Entropy': entro_f[:n],'Margin': margin_f[:n],'Rand': random_f[:n],'Ours':fs3[:n]}
emfpd = pd.DataFrame(data=emf)

n=41
#em = {'#Labels queried':(np.array(margin_r)*2000+1000)[:n],'Badge': bald_em[:n], 'Conf': least_em[:n], \
#      'Entropy': entro_em[:n],'Margin': margin_em[:n],'Rand': random_em[:n],'Ours':em3[:n]}
#empd = pd.DataFrame(data=em)
# palette=sns.color_palette("Paired")[:6]
# # sns.set_style("whitegrid")darkgrid, whitegrid, dark, white, ticks}
# sns.set( style='whitegrid', color_codes=True, rc={"lines.linewidth": 2.2},font_scale = 1.3)
# dffs = emfpd.melt('#Labels queried', var_name='Methods',  value_name='F1')
# gfs = sns.lineplot(x="#Labels queried", y="F1", hue='Methods',palette=palette,  data=dffs)
# gfs.set_title('SQuAD Dataset')

# figfs = gfs.get_figure()
# figfs.show()
# figfs.savefig(os.path.join("./figures","squad_fs_" + str(n)+".pdf"),bbox_inches='tight')


palette=sns.color_palette("Paired")[:6]
# sns.set_style("whitegrid")darkgrid, whitegrid, dark, white, ticks}
sns.set( style='whitegrid', color_codes=True, rc={"lines.linewidth": 2.2},font_scale = 1.3)
df = empd.melt('#Labels queried', var_name='Methods',  value_name='EM')
gem = sns.lineplot(x="#Labels queried", y="EM", hue='Methods',palette=palette,  data=df)
# gem.set_xticklabels(np.concatenate([[0], np.arange(1000,30000,2000)]))
# gem.set_yticklabels(np.concatenate([[0], np.arange(30,80,5)]))
gem.set_title('SQuAD Dataset')

figem = gem.get_figure()
figem.show()
figem.savefig(os.path.join("./figures","squad_em_" + str(n)+".pdf"),bbox_inches='tight')

