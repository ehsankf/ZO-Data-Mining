import numpy as np
import os
import csv
import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns 
import pdb
#import amssymb

def get_data_plot(name_func, path):
    for file in os.listdir(path) :
        if file.find('tfevents') > -1 :
            path = path + '/' + file         
            #example of path is 'example_results/events.out.tfevents.1496749144.L-E7-thalita'
            ea = event_accumulator.EventAccumulator(path, 
                                                size_guidance={ # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                event_accumulator.IMAGES: 4,
                                                event_accumulator.AUDIO: 4,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 1,
                                                 })
            ea.Reload() # loads events from file 
            for function in ea.Tags()['scalars'] :
                if function.find(name_func) > -1 : #to find an approximate name_func 
                    values=[] #empty list
                    steps=[] #empty list
                    for element in (ea.Scalars(function)) :
                        values.append(element.value) #it is a named_tuple, element['value'] is wrong 
                        steps.append(element.step)  
                        

                    return np.array(steps), np.array(values)

def get_csvdata_plot(index, path):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        head_list = next(csvreader)
        print("length: ", csvreader)
        if index < len(head_list):
          values=[] #empty list
          steps=[] #empty list
          for row in csvreader:
              values.append(float(row[index])) #it is a named_tuple, element['value'] is wrong 
              #steps.append(element.step)  
          return np.array(range(1, len(values)+1)), np.array(values)
          


                    
               
    
def plot_train_eval2(index, paths):

    
    sns.set_palette('Set2')
    if not isinstance(paths, list) :
        paths = [paths]
    plt.figure()
    for i, path in enumerate(paths):
        print(path)
        x_scalar, y_scalar = get_csvdata_plot(index, path)
        print(y_scalar)
        plt.plot(x_scalar[0:], y_scalar[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', frameon=False, fontsize=15)
    plt.xlabel('# of steps', fontsize=15)
    plt.ylabel(y_names[index], fontsize=15)
    plt.tight_layout()
    dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[index])))
    #plt.savefig(y_names[index] + 'attack.png')
    
    # plt.show()

def plot_train_eval(x, results, dir_name, index=0):

    
    sns.set_palette('Set2')
    plt.figure()
    for i, y in enumerate(results):
        plt.plot(x[0:], y[0:], colors[i], label=legend[i], alpha=0.9)
       
    plt.legend(loc='best', frameon=False, fontsize=15)
    plt.xlabel(r'$\nu$', fontsize=18)
    plt.ylabel(y_names[index], fontsize=18)
    plt.tight_layout()
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[0])))
    #plt.savefig(y_names[index] + 'attack.png')
    
    # plt.show()
    
    



'''
import seaborn as sns
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);
    plt.show()

line_plot([1,2,3], [1,2,3], 'training', 'test', title='BTC')
'''

def plot_scatter_norm(name_funcx, name_funcy, name_funcz, path):

    sns.set(style='darkgrid')
    sns.set_palette('Set2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    _, x_scalar = get_csvdata_plot(name_funcx, path)
    _, y_scalar = get_csvdata_plot(name_funcy, path)
    _, z_scalar = get_csvdata_plot(name_funcz, path)
        

    corr = np.corrcoef(x_scalar[:2000], y_scalar[:2000])[0][1]
    ax1.scatter(x_scalar[:2000], y_scalar[:2000], color='k', marker='o', alpha=0.5, s=100)
    ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax1.set_xlabel('nuc-norm')
    ax1.set_ylabel('linf-norm')

    corr = np.corrcoef(x_scalar[:2000], z_scalar[:2000])[0][1]
    ax2.scatter(x_scalar[:2000], z_scalar[:2000], color='k', marker='o', alpha=0.5, s=100)
    ax2.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax2.set_xlabel('nuc-norm')
    ax2.set_ylabel('l2-norm')
    plt.savefig(name_funcx+'norm_attack.png')
    plt.show()


def plot_train_eval_conv_curve(x, results, dir_name, x_ticks, y_ticks, y_ticks_label, index=0):

    x_ticks_label = [r'1', r'$1/\epsilon^{1/2}$', r'$1/n^{1/2}$', r'$1/\epsilon^{3/2}$', r'$1 / n^{3/2}$', r'$1/\epsilon^2$', r'$n$']
    # y_ticks_label = [r'$n^{1/2}/\epsilon$', r'$1/\epsilon^{3/2}$', r'$n^{1/2}/\epsilon$', r'$1/\epsilon^{5/3}$', r'$n^{2/3}/\epsilon$', r'$1/\epsilon^2$', r'$n/\epsilon$']
    sns.set_palette('Set2')
    plt.figure(figsize=(10, 6))
    mark = [np.argmin(np.abs(np.array(x) - t)) for t in x_ticks]
    for i, y in enumerate(results):
        plt.plot(x[0:], y[0:], colors[i], label=legend[i], alpha=0.9, mfc='k', marker='o', markevery=mark)
       
    plt.legend(loc='best', frameon=False, fontsize=15)
    plt.xlabel(r'$b$', fontsize=18)
    plt.ylabel(y_names[index], fontsize=18)
    plt.xticks(ticks=x_ticks, labels=x_ticks_label, fontsize=12)
    plt.yticks(ticks=y_ticks, labels=y_ticks_label, fontsize=14)
    plt.tight_layout()
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[0])))
    #plt.savefig(y_names[index] + 'attack.png')
    
    # plt.show()


colors = ['r-', 'g+--', 'b+-', 'b+-.', 'c+-.']  
legend = ['ZO-PSVRG+', 'ZO-PSPIDER+']
y_names = ['SZO Complexity']

n, eps = 2., 0.55
 
def conv_order_svrg(B, b):
    return B / (np.sqrt(b) * eps)  

def conv_order_spider(B, b):
    m = B / b
    if b < 1./ eps ** (1/2.):
        return B /(np.sqrt(b * m)*eps)
    return B / (m * eps)

# SVRG
# Batches = [n, 1./eps, n, 1./eps, n, 1./eps * np.sqrt(1. / eps), n * np.sqrt(n)]
# batches = [1, 1./ eps ** (1/2.), n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n] 
batches = list(np.linspace(1.0, (n ** (1/2.))-0.0001, 10)) + [n ** (1/2.)] + list(np.linspace((n ** (1/2.)) + 0.01, 1./ eps ** (2./3), 10))
Batches = [n] * 10 + [n] + [n] * 10

batches = list(batches) + [n ** (2./3), 1. / eps, n] 
Batches = list(Batches) + [n, 1./eps * np.sqrt(1. / eps), n * np.sqrt(n)]
y_ticks_label_svrg = [r'$1/\epsilon^{7/4}$', r'$n^{3/4}/\epsilon$', r'$1/\epsilon^{5/3}$', r'$n^{2/3}/\epsilon$', r'$1/\epsilon^2$', r'$n/\epsilon$']
order1 = [n / eps, 1 / eps ** (7./4), n ** (3./4) / eps, 1 / eps ** (5./3), n ** (2./3) / eps, 1. / eps ** 2, n / eps]
y_ticks = [x / order1[-1] for x in [n/eps ** (3./4), n ** (3./4) / eps, n / eps ** (2./3), n ** (2./3) / eps, 1. / eps ** 2, n*1./eps]]


Batches = [n, 1./eps, n, 1./eps, n, 1./eps * np.sqrt(1. / eps), n * np.sqrt(n)]
batches = [1, 1./ eps ** (1/2.), n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n] 
# batches = list(np.linspace(1.0, (n ** (1/2.))-0.0001, 10)) + [n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n] 
# Batches = [n] * 10 + [n] * 5

# Batches = [n, 1./eps, n, 1./eps, n, 1./eps * np.sqrt(1. / eps), n * np.sqrt(n)]
# batches = [1, 1./ eps ** (1/2.), n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n]
xticks = [1, 1./ eps ** (1/2.), n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n]


order = [conv_order_spider(B, b) for (B, b) in zip(Batches, batches)]
batches = [10 * x for x in batches]
order_svrg = [x / order[-1] for x in order]
x_ticks = [10 * x for x in [1, 1./ eps ** (1/2.), n ** (1/2.),  1./ eps ** (2./3), n ** (2./3), 1. / eps, n]]
order1 = [n ** (1/2)/eps, 1/eps ** (3/2), n ** (1/2)/eps, 1/eps ** (5/3), n ** (2/3)/eps, 1/eps ** 2, n/eps]
y_ticks = [x / order1[-1] for x in [n ** (1/2)/eps, 1/eps ** (3/2), n ** (1/2)/eps, 1/eps ** (5/3), n ** (2/3)/eps, 1/eps ** 2, n/eps]]
y_ticks_label = [r'$n^{1/2}/\epsilon$', r'$1/\epsilon^{3/2}$', r'$n^{1/2}/\epsilon$', r'$1/\epsilon^{5/3}$', r'$n^{2/3}/\epsilon$', r'$1/\epsilon^2$', r'$n/\epsilon$']
pdb.set_trace()

plot_train_eval_conv_curve(batches, [order_svrg], './', x_ticks, y_ticks, y_ticks_label, index=0)

pdb.set_trace()

scalar_names = ['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf', 'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc']

y_names = ['IS', 'Clean Loss', 'Adversaial Loss', 'Loss Increment', 'Clean Accuracy (%)', 'Accuracy (%)', r'$L_{\infty}$', r'$L_{0}$', r'$L_{2}$', 'Nuclear Norm']

colors = ['ro-', 'g+--', 'b+-', 'b+-.', 'c+-.']


#path = ['summary/sum_pgd/pgditer7eps0.3SmallCNNmnist.csv', 'summary/sum_pgd/pgditer20eps0.3SmallCNNmnist.csv',
#         'summary/sum_pgd/pgditer41eps0.3SmallCNNmnist.csv']

#path = ['summary/sum_pgd/pgditer7eps0.3Netmnist_Net.csv', 'summary/sum_pgd/pgditer20eps0.3Netmnist_Net.csv',
#         'summary/sum_pgd/pgditer41eps0.3Netmnist_Net.csv']

import torch

path = ['summary/sum_rand_cross_FW_ResNet18/FWiter20eps1.0ResNet18data.csv_dict.pt', 'summary/sum_rand_cross_FW_group_ResNet18/FW_groupiter20eps1.0ResNet18data.csv_dict.pt', 'summary/sum_rand_cross_FW_group_ResNet18/FW_groupiter20eps3.0ResNet18data.csv_dict.pt',
'summary/sum_rand_cross_StrAttack_ResNet18/StrAttackiter10eps-1.0ResNet18data.csv_dict.pt', 'summary/sum_rand_cross_CS_ResNet18/CSiter20eps-1ResNet18data.csv_dict.pt']

legend = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl-group $\epsilon_{S1} = 1$', r'FWnucl-group $\epsilon_{S1} = 3$', 'StrAttack', 'CS']

results_sum = []
for pt in path:
   pt_f = torch.load(pt)
   results_sum.append(pt_f['IS_values'])

'''
scalar_names = ['loss_clean', 'loss_adv']

paths = ['pgd40NetPureMENETNet56']


for name in scalar_names:
    plot_train_eval (name, paths)

plot_scatter_norm('eval_x_adv_lnuc', 'eval_x_adv_linf', 'eval_x_adv_l2', 'pgd40Netmnistadvtensor85acctensor96')

'''
#path = 'summary/sum_FW10eps10.0NetmnistItr40advtensor88.csv'

#path = 'summary/sum_FW10eps10.0NetmnistItr40advtensor88.csv'

#row_index, steps = get_csvdata_plot('eval_x_adv_lnuc', path)

#print(row_index)
#print(steps)

plot_train_eval(np.arange(30, 100, 10), results_sum, dir_name = path[0].split('/')[0])
#for i in range(len(scalar_names)):
#    plot_train_eval(i, path)



