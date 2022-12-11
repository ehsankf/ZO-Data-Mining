## Universal_Attack.py -- The main entry file for attack generation
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
##                     Chun-Chen Tu <timtu@umich.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import sys
sys.path.append('models/')
sys.path.append('optimization_methods/')

import os
import numpy as np
import argparse

from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
import Utils as util
import ObjectiveFunc
import ZO_SVRG as svrg
import ZO_SPIDER as spider
import ZO_SAGA as saga
import ZO_SGD as sgd
from SysManager import SYS_MANAGER

import pdb

MGR = SYS_MANAGER()
# command line
# python Universal_Attack.py -optimizer ZOSGD
# nohup python Universal_Attack.py -optimizer ZOPSVRGRSGE -dataset cifar -nStage 100 > log_ZOPSVRGRSGE&
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = MGR.parSet['gpu_id']
    if(MGR.parSet['dataset'] == 'mnist'):
        data, model =  MNIST(), MNISTModel(restore="models/mnist", use_log=True)
    elif(MGR.parSet['dataset'] == 'cifar'):
        data, model =  CIFAR(), CIFARModel(restore="models/cifar", use_softmax=True)
    origImgs, origLabels, origImgID = util.generate_attack_data_set(data, model, MGR)

    delImgAT_Init = np.zeros(origImgs[0].shape)
    # delImgAT_Init = np.random.randn(*origImgs[0].shape) * 0.01
    objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels) 

    MGR.Add_Parameter('eta', MGR.parSet['alpha']/np.sqrt(origImgs[0].size)) #MGR.parSet['alpha']/np.sqrt(origImgs[0].size)
    MGR.Log_MetaData()

    if(MGR.parSet['optimizer'] == 'ZOSVRG'):
        MGR.parSet['rv_dist'] = 'CoordSphere'
        MGR.parSet['B_size'] = 0.90
        MGR.parSet['alpha'] = 0.005
        MGR.Add_Parameter('eta', MGR.parSet['alpha'])
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = svrg.ZOSVRG(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOPSVRG'):
        MGR.parSet['rv_dist'] = 'CoordSphere'
        MGR.parSet['B_size'] = 0.50
        MGR.parSet['alpha'] = 0.005
        MGR.Add_Parameter('eta', MGR.parSet['alpha'])
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = svrg.ZOSVRG(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOPSPIDER'):
        MGR.parSet['rv_dist'] = 'CoordSphere'
        MGR.parSet['B_size'] = 0.50
        MGR.parSet['alpha'] = 0.003
        MGR.Add_Parameter('eta', MGR.parSet['alpha'])
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = spider.ZOSPIDER(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOSVRGRSGE'):
        MGR.parSet['rv_dist'] = 'UnitSphere'
        MGR.parSet['qx'] = 1
        MGR.parSet['qy'] = 1
        MGR.parSet['B_size'] = 0.9
        MGR.parSet['alpha'] = 0.005
        MGR.Add_Parameter('eta', MGR.parSet['alpha']/np.sqrt(origImgs[0].size))
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = svrg.ZOSVRG(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOPSVRGRSGE1'):
        MGR.parSet['rv_dist'] = 'UnitSphere'
        MGR.parSet['qx'] = 1
        MGR.parSet['qy'] = 1
        MGR.parSet['B_size'] = 0.5
        MGR.parSet['alpha'] = 0.004
        MGR.Add_Parameter('eta', MGR.parSet['alpha']/np.sqrt(origImgs[0].size))
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = svrg.ZOSVRG(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOPSPIDERRSGE'):
        MGR.parSet['rv_dist'] = 'UnitSphere'
        MGR.parSet['qx'] = 1
        MGR.parSet['qy'] = 1
        MGR.parSet['B_size'] = 0.50
        MGR.parSet['alpha'] = 0.005
        MGR.Add_Parameter('eta', MGR.parSet['alpha']/np.sqrt(origImgs[0].size))
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = spider.ZOSPIDER(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOSAGA'):
        MGR.parSet['rv_dist'] = 'CoordSphere'
        MGR.parSet['B_size'] = 1.0
        MGR.parSet['alpha'] = 0.01 # 0.005
        MGR.Add_Parameter('eta', MGR.parSet['alpha']/np.sqrt(origImgs[0].size))
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = saga.ZOSAGA(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOSGD'):
        MGR.parSet['rv_dist'] = 'CoordSphere'
        MGR.parSet['B_size'] = 1.0
        MGR.parSet['alpha'] = 0.10
        MGR.Add_Parameter('eta', MGR.parSet['alpha']/origImgs[0].size)
        objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)
        delImgAT = sgd.ZOSGD(delImgAT_Init, MGR, objfunc)
    else:
        print('Please specify a valid optimizer')


    for idx_ImgID in range(MGR.parSet['nFunc']):
        currentID = origImgID[idx_ImgID]
        orig_prob = model.model.predict(np.expand_dims(origImgs[idx_ImgID], axis=0))
        advImg = np.tanh(np.arctanh(origImgs[idx_ImgID]*1.9999999)+delImgAT)/2.0
        adv_prob  = model.model.predict(np.expand_dims(advImg, axis=0))

        suffix = "id{}_Orig{}_Adv{}".format(currentID, np.argmax(orig_prob), np.argmax(adv_prob))
        util.save_img(advImg, "{}/Adv_{}.png".format(MGR.parSet['save_path'], suffix))
    util.save_img(np.tanh(delImgAT)/2.0, "{}/Delta.png".format(MGR.parSet['save_path']))

    sys.stdout.flush()
    MGR.logHandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset' , default='mnist', help="choose from ZOSVRG and ZOSGD")
    parser.add_argument('-optimizer' , default='ZOSVRG', help="choose from ZOSVRG and ZOSGD")
    #parser.add_argument('-qx', type=int, default=10, help="Number of random vectors to average over for each gradient estimation")
    #parser.add_argument('-qy', type=int, default= 10, help="Number of random vectors to average over for each gradient estimation")
    parser.add_argument('-qx', type=int, default=28, help="Number of random vectors to average over for each gradient estimation")
    parser.add_argument('-qy', type=int, default=28, help="Number of random vectors to average over for each gradient estimation")
    parser.add_argument('-alpha', type=float, default= 0.2, help="Optimizer's step size being (alpha)/(input image size)")
    parser.add_argument('-M', type=int, default=10, help="Length of each stage/epoch")
    parser.add_argument('-nStage', type=int, default=3000, help="Number of stages/epochs")
    parser.add_argument('-const', type=float, default=5, help="Weight put on the attack loss")
    parser.add_argument('-lambda2', type=float, default=1.0, help="Coefficeint on L2 regularizer")
    parser.add_argument('-lambda1', type=float, default=1.0, help="Coefficeint on L1 regularizer")
    parser.add_argument('-nFunc', type=int, default=10, help="Number of images being attacked at once")
    parser.add_argument('-B_size', type=float, default=1, help="Fraction number of functions sampled for each iteration in the optmization steps")
    parser.add_argument('-batch_size', type=int, default=5, help="Number of functions sampled for each iteration in the optmization steps")
    parser.add_argument('-mu', type=float, default=0.001, help="The weighting magnitude for the random vector applied to estimate gradients in ZOSVRG")
    #parser.add_argument('-rv_dist', default='UnitSphere', help="Choose from UnitSphere and UnitBall")
    parser.add_argument('-rv_dist', default='UnitSphere', help="Choose from UnitSphere and UnitBall")
    parser.add_argument('-target_label', type=int, default=1, help="The target digit to attack")
    parser.add_argument('-gpu_id', type=str, default="0", help="The gpu number")
    args = vars(parser.parse_args())

    for par in args:
        MGR.Add_Parameter(par, args[par])

    MGR.Add_Parameter('save_path', 'Results/' + MGR.parSet['optimizer'] + '/')
    MGR.parSet['batch_size'] = min(MGR.parSet['batch_size'], MGR.parSet['nFunc'])

    main()
