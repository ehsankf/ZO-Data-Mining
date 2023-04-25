## ZO_SVRG.py -- Perform ZOSVRG Optimization Algorithm
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
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

import numpy as np

np.random.seed(2018)

def ZOSAGA(delImgAT_Init, MGR, objfunc):

    best_Loss = 1e10
    best_delImgAT = delImgAT_Init
    #grad_avg = np.zeros(delImgAT_Init.shape)
    matrix_M = np.zeros([delImgAT_Init.shape[0], delImgAT_Init.shape[0], MGR.parSet['nFunc']])
    print('delImgAT_Init.shape: ', delImgAT_Init.shape, 'phase', MGR.parSet['nFunc'])
    g_S = np.zeros(delImgAT_Init.shape)      

    delImgAT_kp1_S = delImgAT_Init
    for S_idx in range(1, MGR.parSet['nStage']+1):

    #delImgAT_M_Sm1 = delImgAT_kp1_S
    #randBatchBdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), int(MGR.parSet['B_size']*MGR.parSet['nFunc']), replace=False)
    #print(randBatchBdx)
    #print("\n")
    #g_S = objfunc.gradient_estimation(delImgAT_M_Sm1, MGR.parSet['mu'], MGR.parSet['qx'], MGR.parSet['qy'], randBatchBdx)
    #delImgAT_0_S = delImgAT_M_Sm1

    #objfunc.evaluate(delImgAT_0_S, np.array([]), False)

    #delImgAT_kp1_S = delImgAT_0_S
    #for k in range(0, MGR.parSet['nStage']+1):
      for k in range(0, MGR.parSet['M']):
            delImgAT_k_S = delImgAT_kp1_S
            
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), MGR.parSet['batch_size'], replace=False)
            batchSize = randBatchIdx.size
            h_k_S = np.zeros(delImgAT_Init.shape)
            for i in randBatchIdx:
                h_k_S[:,:,0] += matrix_M[:, :, i]
            h_k_S *= 1/batchSize
            v_k_S  = objfunc.gradient_estimation(delImgAT_k_S, MGR.parSet['mu'], MGR.parSet['qx'], MGR.parSet['qy'], randBatchIdx)
            matrix_M[:, :, randBatchIdx] = v_k_S

            #v_k_S  *= batchSize
            
            '''
            for (j = 0; j < d; j ++){
                                 g_tilda[j] = Matrix[train_index][j];
				//g_tilda[j] = Matrix[idx][j];
                                //g_prev[j] += lambda * w_prev[j];
		       	}
            for (j = 0; j < d; j ++){
                             mu[j] += (g_prev[j] - g_tilda[j])/n;
			    // mu[j] += lambda * w_tilda[j];
                             
		        }
            for (j = 0; j < d; j ++){
				g_prev[j] += lambda * w_prev[j];
				//g_tilda[j] += lambda * w_tilda[j]; 
                              //  if(w_prev[j] >= 0)
                             //        g_prev[j] += lambda1;      
                              //  else
                             //        g_prev[j] -= lambda1;
			}

			// Update the test point 
			update_test_point_SVRG(w_prev, g_prev, g_tilda, mu, alpha, d);


                       /* double param = alpha * 1e-6;
                        for (j = 0; j < d; j ++){
                        if(w_prev[j] > param)
                           w_prev[j] -= param;
                        else if(w_prev[j] < -param)
                           w_prev[j] += param;
                        else
                            w_prev[j] = 0;
                        }*/ 
            '''
            
 
            g_S += (v_k_S - h_k_S)/MGR.parSet['nFunc'];
            
            v_k_S -=h_k_S;
            
            #v_k_S -= objfunc.gradient_estimation(delImgAT_0_S, MGR.parSet['mu'], MGR.parSet['qx'], MGR.parSet['qy'], randBatchIdx)
            v_k_S += g_S
            
            delImgAT_kp1_S = delImgAT_k_S - batchSize * MGR.parSet['eta'] * v_k_S
            
            objfunc.evaluate(delImgAT_kp1_S, np.array([]), False)
            if((S_idx*MGR.parSet['M']+k)%100 == 0):
                    print('Stage Index: ', S_idx, '       M Index: ', k)
                    objfunc.print_current_loss()
            if(objfunc.Loss_Attack <= 1e-20 and objfunc.Loss_Overall < best_Loss):
                best_Loss = objfunc.Loss_Overall
                best_delImgAT = delImgAT_kp1_S
                #print('Updating best delta image record')

            MGR.logHandler.write('S_idx: ' + str(S_idx))
            MGR.logHandler.write(' m_idx: ' + str(k))
            MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
            MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
            MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
            MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
            MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
            MGR.logHandler.write('\n')

    return best_delImgAT
