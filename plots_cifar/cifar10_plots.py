
import matplotlib.pyplot as plt
import numpy as np 
import os
import pdb

def clean_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid("off")

    
# nrows = len(epsilons)
# ncols = num_display

algos = ['Original\nImages', 'ZO-ProxSVRG', 'ZO-PSVRG+', 'ZO-PSPIDER+', 'ZO-PSVRG+\n(RandSGE)', 
         'ZO-ProxSAGA', 'ZO-ProxSGD']

all_pathes = ['cifar', 'ZOSVRG', 'ZOPSVRG', 'ZOPSPIDER', 'ZOPSVRGRSGE1', 'ZOSAGA', 'ZOSGD1']
# all_pathes = ['ZOSGD', 'ZOSVRG', 'ZOPSVRGRSGE', ]

all_adv_examples = ['../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG/Delta.png.npy', 
        '../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG/Delta.png.npy', 
        '../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG/Delta.png.npy',
        '../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD/Delta.png.npy']

all_adv_examples = []
for path in [all_pathes[0]]:
    for file in os.listdir(os.path.join('Results', path)):
        if file.endswith(".npy") and not file.startswith('Delta'):
            all_adv_examples.append(file.split('Orig')[0])
    all_adv_examples.append('Delta.png.npy')


nrows = len(all_pathes) # 1
ncols = len(all_adv_examples) + 1 # len(all_adv_examples)



'''
annots = ['Algorithm']
for col in range(1):
    axis = fig.add_subplot(spec[0, col]) # ax[0, col] # fig.add_subplot(spec[0, col])
    axis.annotate(annots[col], (0.5, 0.5),
                          transform=axis.transAxes,
                          ha='center', va='center', fontsize=18,
                          color='darkgrey')
    axis.axis("off")


axis = fig.add_subplot(spec[0, 2:4])
axis.annotate("Perturbed Images", (0.5, 0.5),
                      transform=axis.transAxes,
                      ha='center', va='center', fontsize=18,
                      color='darkgrey')
axis.axis("off")

axis = fig.add_subplot(spec[0, 4])
axis.annotate(r"Distortion ($\delta$)", (0.5, 0.5),
                      transform=axis.transAxes,
                      ha='center', va='center', fontsize=18,
                      color='darkgrey')
axis.axis("off")
'''
# fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
fig = plt.figure(figsize=(ncols-1, nrows),)
spec = fig.add_gridspec(nrows, ncols, width_ratios=[1] *ncols)
# spec.update(wspace=0.025, hspace=0.05) 
spec.update(wspace=-0.35,) 
for row in range(nrows):
    # if row > 0:
    files_in_path = os.listdir(os.path.join('Results', all_pathes[row]))
    for col in range(ncols):
        axis = fig.add_subplot(spec[row, col]) # ax[row, col]
        # axis = ax[row, col]
        axis.axis("off")
        
        if row == 0 and col == ncols - 2:
            axis.annotate(' Universal\n Perturbation\n' + r'($\delta$)', (0.5, 0.5),
                          ha='center', va='center', fontsize=8,
                          color='black')
            continue
  
        if row == 0 and col == ncols - 1:
            axis.annotate('Distorsion\nNorm\n' + r'($\|\|\delta\|\|$)', (0.6, 0.5),
                         ha='center', va='center', fontsize=8,
                         color='black')
            continue
   
        """
        if row == 0:
            continue
        """
        if col == 0:
            axis.annotate(algos[row], (0.3, 0.5),
                          ha='center', va='center', fontsize=10,
                          color='black')
            continue
        if col == ncols - 1:
            perturbed_image = np.load(os.path.join('Results', all_pathes[row], 'Delta.png.npy'))
            perturbed_image = np.around((perturbed_image + 0.5)*255)
            l2_norm_delta = np.linalg.norm(perturbed_image.flatten(), ord=2)/255.
            axis.annotate(f'{l2_norm_delta:.2f}', (0.6, 0.5),
                          ha='center', va='center', fontsize=12,
                          color='black')
            continue
        label, perturbed_pred = 1, 1 
        filter_perturbed_image = filter(lambda x: x.startswith(all_adv_examples[col]), files_in_path)
        for filter_f in filter_perturbed_image:
            if filter_f.endswith('npy'):
                perturbed_image = np.load(os.path.join('Results', all_pathes[row], filter_f))
        perturbed_image = np.around((perturbed_image + 0.5)*255)
        perturbed_image = perturbed_image.astype(np.uint8).squeeze()
        axis.imshow(perturbed_image, interpolation='nearest')
        if not (row == 0 or col >= ncols-2):
            label, perturbed_pred = 1, filter_f.split('Adv')[2].split('.')[0]
            axis.set_title(f"{label} -> {perturbed_pred}", fontsize=8, y=-.27)
        clean_axis(axis)
        # if col == 0:
            #axis.set_ylabel(f"Epsilon: {epsilons[row]}")
plt.tight_layout()
# plt.show()
plt.pause(1.5)
plt.savefig("{}.png".format("cifar"), bbox_inches='tight', dpi=300)
plt.close()

