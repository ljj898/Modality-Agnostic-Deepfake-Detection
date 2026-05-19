"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
from tqdm import tqdm
from .metrics import *
#from .metrics import compute_cer, compute_wer, evaluate_auc, confusion_matrix, get_acc, get_eer
import ipdb
import pdb
import torch.nn as nn
#from .decoders import ctc_greedy_decode, ctc_search_decode
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score

def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad==True])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function,fakeNumloss, device, trainParams, global_label):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    loss1=0
    loss2=0
    
    if global_label==False:
        for batch, (inputBatch, inputLenBatch, labelBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                            ncols=75)):
    #input[0] torch.Size([1008, 12, 321]); input[1] torch.Size([252, 12, 512])
            inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device))
            inputLenBatch = (inputLenBatch.int()).to(device)
            #pdb.set_trace()
            #label batch(lenth,Batch)->(Batch,length)
            
            labelBatch = (labelBatch.long()).to(device).transpose(0,1)
            
            opmode = np.random.choice(["AO", "VO", "AV"],
                                    p=[trainParams["aoProb"], trainParams["voProb"], 1-(trainParams["aoProb"]+trainParams["voProb"])])
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                pass
            
            optimizer.zero_grad()
            model.train()
            ## outputBatch torch.Size([283, 12, 2])->torch.Size([12, 2, 283])
            outputBatch = model(inputBatch).transpose(1,2).transpose(0,2)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, labelBatch)
            loss.backward()
            optimizer.step()

            trainingLoss = trainingLoss + loss.item()
    else:
        for batch, (inputBatch, inputLenBatch, labelBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                            ncols=75)):
    #input[0] torch.Size([1008, 12, 321]); input[1] torch.Size([252, 12, 512])
            inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device))
            inputLenBatch = (inputLenBatch.int()).to(device)      
            labelBatch = (labelBatch.long()).to(device)
            
            opmode = np.random.choice(["AO", "VO", "AV"],
                                    p=[trainParams["aoProb"], trainParams["voProb"], 1-(trainParams["aoProb"]+trainParams["voProb"])])
            
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
                labelBatch[:,0]=-1
                fakeBatch=labelBatch[:,1]
                #print(fakeBatch.shape)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
                labelBatch[:,1]=-1
                fakeBatch=labelBatch[:,0]
                #print(fakeBatch.shape)
            else:
                fakeBatch=torch.sum(labelBatch,1)
                #print(fakeBatch.shape)
            
            optimizer.zero_grad()
            model.train()
            ## outputBatch torch.Size([283, 12, 2])->torch.Size([12, 2, 283])
            #pdb.set_trace()
            
            outputBatch,fakNumBatch = model(inputBatch, inputLenBatch)#.transpose(1,2).transpose(0,2)
            batch_size = outputBatch.size()[0]
            seq_len = outputBatch.size()[1]
            
            with torch.backends.cudnn.flags(enabled=False):
                #pdb.set_trace()
                #outbatch[12,2] and labelbatch[12,2]
                #
                tmp_loss = loss_function( outputBatch.view(batch_size * seq_len, 1),labelBatch.view(batch_size * seq_len, 1).float())
                #pdb.set_trace()
                tmp_mask = (labelBatch != -1)
                real_loss = tmp_loss.view(batch_size, seq_len) * tmp_mask
                clsloss = torch.sum(real_loss) / torch.sum(tmp_mask)
                #clsloss = loss_function(outputBatch, labelBatch.float())
                #clsloss = loss_function(torch.unsqueeze(outputBatch,1), labelBatch)
                if fakNumBatch is not None:
                    fakeloss = fakeNumloss(fakNumBatch,fakeBatch)
                else:
                    fakeloss=0.0
                loss=clsloss+fakeloss
            
            loss.backward()
            optimizer.step()
            loss1 = loss1 + clsloss.item()
            if fakNumBatch is not None:
                loss2 = loss2 + fakeloss.item()
            trainingLoss = trainingLoss + loss.item()
    loss1 = loss1/len(trainLoader) 
    loss2 = loss2/len(trainLoader) 
    trainingLoss = trainingLoss/len(trainLoader) 
     
    return loss1,loss2,trainingLoss


def scores_evaluation(scores_, targets_):
    print('evaluation start...')
    scores_=np.array(scores_)
    targets_=np.array(targets_)
    n, n_class = np.shape(scores_)
    print('img_numbers=',n,'n_class=',n_class)
    Nc, Np, Ng, Nn,Ntn = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class),np.zeros(n_class), np.zeros(n_class)
    cls_P,cls_R,cls_F1,cls_Acc=np.zeros(n_class),np.zeros(n_class),np.zeros(n_class),np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]         # all img scores on class_k
        targets = targets_[:, k]       # all img labels on class_k
        
        targets[targets == -1] = 0     # set img labels from -1 to 0
        Ng[k] = np.sum(targets == 1)   # ture:     all ture positive labels sum number
        Np[k] = np.sum(scores >= 0)    # positive: all predict positive sum number
        Nn[k] = np.sum(scores < 0)     # negative: all predict negative sum number
        
        Nc[k] = np.sum(targets * (scores >= 0)) # true_positive: true_positive sum number
        Ntn[k] = np.sum((1-targets) * (scores < 0)) # true_positive: true_negative sum number
        cls_P[k]=Nc[k]/Np[k]
        cls_R[k]=Nc[k]/Ng[k]
        cls_F1[k]=(2 * cls_P[k] * cls_R[k]) / (cls_P[k] + cls_R[k])
        
        cls_Acc[k]=(Nc[k] +  Ntn[k])/targets.shape[0]
    Np[Np == 0] = 1
    print('np.sum(Nc),true_positive=',np.sum(Nc))
    print('np.sum(Np),positive=',np.sum(Np))
    print('np.sum(Ng),ture=',np.sum(Ng))
 
    # for all labels num_imgs*n_classes
    OP = np.sum(Nc) / np.sum(Np)        # precision: true_positive/positive
    OR = np.sum(Nc) / np.sum(Ng)        # recall:    true_positive/true
    OF1 = (2 * OP * OR) / (OP + OR)     # F1_score: harmonic mean of precision and recall
    # average by class
    CP = np.sum(Nc / Np) / n_class      # precision: true_positive/positive
    CR = np.sum(Nc / Ng) / n_class      # recall:    true_positive/true
    CF1 = (2 * CP * CR) / (CP + CR)     # F1_score: harmonic mean of precision and recall
 
    return OP, OR, OF1, CP, CR, CF1,cls_P,cls_R,cls_F1,cls_Acc

def evaluate(model, evalLoader, step, device,evalParams, global_label):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """
    all_outputs =[]
    all_outsigmoid=[]
    all_preds = []
    all_labels = []
    all_audlabels = []
    all_vidlabels = []
    all_pos_scores = []
    all_predsA = []
    all_predsV = []
    all_predsB=[]
    all_labelsB=[]
    results = []
    if global_label==False:
        for batch, (inputBatch, inputLenBatch, labelBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                            ncols=75)):
        
            inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device))
            #torch.Size([252, 12, 1])
            inputLenBatch = (inputLenBatch.int()).to(device)#.transpose(0,1)#, (targetLenBatch.int()).to(device)
            labelBatch = (labelBatch.long()).to(device)#.transpose(0,1)
            opmode = np.random.choice(["AO", "VO", "AV"],
                                     p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                 pass

            model.eval()
            with torch.no_grad():
                outputBatch = model(inputBatch)#torch.Size([252, 12, 2])
            outputmask=torch.mul(inputLenBatch,outputBatch)
            valid_lenth=torch.sum(inputLenBatch,0)#torch.Size([12, 1])
            pre_video=torch.sum(outputmask,0)/valid_lenth#torch.Size([12, 2])
            
            _, predicted = torch.max(pre_video, 1)
            #pdb.set_trace()
            outputs =pre_video[:, -1].view(-1)
            all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labelBatch.cpu().numpy().tolist())
    else:
        for batch, (inputBatch, inputLenBatch, labelBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                            ncols=75)):
        
            inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device))
            inputLenBatch = (inputLenBatch.int()).to(device)#.transpose(0,1)#, (targetLenBatch.int()).to(device)
            labelBatch = (labelBatch.long()).to(device)#.transpose(0,1)
            opmode = np.random.choice(["AO", "VO", "AV"],
                                     p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                pass

            model.eval()
            with torch.no_grad():
                outputBatch,fakNumBatch = model(inputBatch,inputLenBatch)#.transpose(1,2).transpose(0,2)
           
            #pdb.set_trace()
            #post_function = nn.Softmax(dim=1)
            #outputBatch = post_function(outputBatch)
            
            #_, predicted = torch.max(outputBatch , 1)
            predictedA = torch.sigmoid(outputBatch[:,1])
            predictedV = torch.sigmoid(outputBatch[:,0])
            
            predictedB =1-((1-predictedA)*(1-predictedV))
            AudioLabels=labelBatch[:,1]
            VideoLabels=labelBatch[:,0]
            BLabels=torch.bitwise_or(AudioLabels, VideoLabels)
          
            all_outputs.extend(outputBatch.detach().cpu().numpy())
            all_outsigmoid.extend(torch.sigmoid(outputBatch).detach().cpu().numpy())
            #all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
         
            all_labels.extend(labelBatch.cpu().numpy())
            all_predsA.append(predictedA.cpu().numpy().tolist())
            all_predsV.append(predictedV.cpu().numpy().tolist())
            all_predsB.append(predictedB.cpu().numpy().tolist())
            all_audlabels.append(AudioLabels.cpu().numpy().tolist())
            all_vidlabels.append(VideoLabels.cpu().numpy().tolist())
            all_labelsB.append(BLabels.cpu().numpy().tolist())
    #ipdb.set_trace()
    #print('np.shape(total_results)',np.shape(all_outputs))
    #print('np.shape(total_labels)',np.shape(all_labels))
    OP,OR,OF1,CP,CR,CF1,cls_P,cls_R,cls_F1,cls_Acc=scores_evaluation(scores_=all_outputs,targets_=all_labels)
    
    #pdb.set_trace() 
    Vroc_auc = roc_auc_score(all_vidlabels, all_predsV)
    Aroc_auc = roc_auc_score(all_audlabels, all_predsA)
    
    Broc_auc =roc_auc_score(all_labelsB, all_predsB)

    #pdb.set_trace()
    prediction_int = np.zeros_like(np.array(all_outsigmoid))
    prediction_int[np.array(all_outputs) > 0.5] = 1
    pdb.set_trace()
    Vbacc, Vroc_auc = evaluate_auc(all_vidlabels, prediction_int.astype(np.int32)[:, 0], all_predsV)
    Abacc, Aroc_auc = evaluate_auc(all_audlabels, prediction_int.astype(np.int32)[:, 1], all_predsA)
    result = np.bitwise_or(prediction_int.astype(np.int32)[:, 0], prediction_int.astype(np.int32)[:, 1])
    Bacc = get_acc(all_labelsB, result)
    #pdb.set_trace()
    #print('宏平均F1-score:',metrics.f1_score(all_labels,prediction_int,average='macro'))#预测宏平均f1-score输出
    #print('微平均F1-score:',metrics.f1_score(all_labels,prediction_int,average='micro'))#预测微平均f1-score输出
    #print('加权平均F1-score:',metrics.f1_score(all_labels,prediction_int,average='weighted'))
    WP=metrics.precision_score(all_labels,prediction_int, average='weighted')
    WR=metrics.recall_score(all_labels,prediction_int, average='weighted')
    WF1=metrics.f1_score(all_labels,prediction_int,average='weighted')
    all_evaluate_results={'BACC':Bacc,'OP':OP,'OR':OR,'OF1':OF1,'CP':CP,'CR':CR,'CF1':CF1,'cls_P':cls_P,'cls_R':cls_R,'cls_F1':cls_F1}
    weight_results={'STEP':step,'VAUC':Vroc_auc, 'AAUC':Aroc_auc,'BAUC':Broc_auc,'WP':WP,'WR':WR,'WF1':WF1,'cls_Acc':cls_Acc}

    # acc = get_acc(all_labels, all_preds)
    # bacc, roc_auc = evaluate_auc(all_labels, all_preds, all_pos_scores)
    # TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
    # real_recall = TN / (TN + FP)
    # fake_recall = TP / (TP + FN)
    # eer = get_eer(all_labels, all_pos_scores)

    # far = FP / (FP + TN)
    # frr = FN / (FN + TP)
    # hter = (far + frr) / 2
    # result = 'model:{},Total images:{},acc:{:.6f},bACC:{:.6f},RR:{:.6f},FR:{:.6f},ROC_AUC:{:.6f},EER:{:.6f},' \
    #          'HTER:{:.6f},TN:{},FN:{},TP:{},FP:{}' \
    #     .format(str(step), len(all_labels), acc, bacc, real_recall, fake_recall, roc_auc, eer, hter, TN, FN, TP, FP)
    print(step,all_evaluate_results,weight_results)
    #results.append(str(all_evaluate_results) + '\n')    
    return all_evaluate_results,weight_results
