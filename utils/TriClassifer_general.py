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
import torch.nn as nn
#from .decoders import ctc_greedy_decode, ctc_search_decode
from sklearn import metrics

def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad==True])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function,CEloss, device, trainParams, global_label):

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
           
            
            optimizer.zero_grad()
            model.train()
            ## outputBatch torch.Size([283, 12, 2])->torch.Size([12, 2, 283])
            outputBatch,outA ,outV = model(inputBatch).transpose(1,2).transpose(0,2)
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
            
            
           
            
            optimizer.zero_grad()
            model.train()
            ## outputBatch torch.Size([283, 12, 2])->torch.Size([12, 2, 283])
            #pdb.set_trace()
            
            outputBatch,outA ,outV = model(inputBatch, inputLenBatch)#.transpose(1,2).transpose(0,2)
            batch_size = outputBatch.size()[0]
            seq_len = outputBatch.size()[1]
            
            with torch.backends.cudnn.flags(enabled=False):
                #pdb.set_trace()
                #outbatch[12,2] and labelbatch[12,2]
                #
              
                Aloss=CEloss(outA,labelBatch[:,1])
                Vloss=CEloss(outV,labelBatch[:,0])
                # import pdb
                # pdb.set_trace()
                Jlabel=torch.logical_or(labelBatch[:,1],labelBatch[:,0]).int().type(torch.int64)
                Jloss=CEloss(outputBatch,Jlabel)
                #clsloss = loss_function(outputBatch, labelBatch.float())
                #clsloss = loss_function(torch.unsqueeze(outputBatch,1), labelBatch)
                
                loss=Aloss+Vloss+Jloss
            
            loss.backward()
            optimizer.step()
            loss1 = loss1 + Aloss.item()
            loss2 = loss2 + Vloss.item()
            
            trainingLoss = trainingLoss + Jloss.item()
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
    all_preds = []
    all_labels = []
    all_pos_scores = []
    all_predsA = []
    all_audlabels = []
    all_pos_scoresA = []
    all_predsV = []
    all_vidlabels = []
    all_pos_scoresV = []
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
                output,outputA ,outputV = model(inputBatch, inputLenBatch)#.transpose(1,2).transpose(0,2)
           
            #pdb.set_trace()
            #post_function = nn.Softmax(dim=1)
            #outputBatch = post_function(outputBatch)
            
            
            _, predicted = torch.max(output,1)
            _, predictedA = torch.max(outputA,1)
            _, predictedV = torch.max(outputV,1)
            EntireLabels=torch.logical_or(labelBatch[:,1],labelBatch[:,0]).int().type(torch.int64)
            AudioLabels=labelBatch[:,1]
            VideoLabels=labelBatch[:,0]
            #pdb.set_trace()
            outputs =output[0][-1].view(-1)
            outputsA =outputA[0][-1].view(-1)
            outputsV =outputV[0][-1].view(-1)
            #print (outputBatch.shape)[12,2]
            # import pdb
            # pdb.set_trace()
            #all_outputs.extend(output.detach().cpu().numpy())
            #all_outsigmoid.extend(torch.sigmoid(output).detach().cpu().numpy().tolist())
            all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
            all_pos_scoresA.extend(outputsA.detach().cpu().numpy().tolist())
            all_pos_scoresV.extend(outputsV.detach().cpu().numpy().tolist())
            #all_preds.extend(predicted.cpu().numpy().tolist())
            all_preds.append(predicted.cpu().numpy().tolist())
            all_predsA.append(predictedA.cpu().numpy().tolist())
            all_predsV.append(predictedV.cpu().numpy().tolist())
            all_labels.append(EntireLabels.cpu().numpy().tolist())
            all_audlabels.append(AudioLabels.cpu().numpy().tolist())
            all_vidlabels.append(VideoLabels.cpu().numpy().tolist())
    #ipdb.set_trace()
    #print('np.shape(total_results)',np.shape(all_outputs))
    #print('np.shape(total_labels)',np.shape(all_labels))
    #OP,OR,OF1,CP,CR,CF1,cls_P,cls_R,cls_F1,cls_Acc=scores_evaluation(scores_=all_outputs,targets_=all_labels)
    
    # import pdb
    # pdb.set_trace()
    acc = get_acc(all_labels, all_preds)
    F1  = get_f1(all_labels, all_preds)
    bacc, roc_auc = evaluate_auc(all_labels, all_preds, all_pos_scores)
    TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
    real_recall = TN / (TN + FP)
    fake_recall = TP / (TP + FN)
    eer = get_eer(all_labels, all_pos_scores)

    far = FP / (FP + TN)
    frr = FN / (FN + TP)
    hter = (far + frr) / 2
    
    Aacc = get_acc(all_audlabels, all_predsA)
    AF1  = get_f1(all_audlabels, all_predsA)
    # if len(numpy.unique(all_audlabels))<=1:Abacc, Aroc_auc,Aeer = float('-inf'), float('-inf'), float('-inf')
    # else: 
    Abacc, Aroc_auc = evaluate_auc(all_audlabels, all_predsA, all_pos_scoresA)
    Aeer = get_eer(all_audlabels, all_pos_scoresA)
    ATN, AFP, AFN, ATP, = confusion_matrix(all_audlabels, all_predsA).ravel()
    Areal_recall = ATN / (ATN + AFP)
    Afake_recall = ATP / (ATP + AFN)
    

    Afar = AFP / (AFP + ATN)
    Afrr = AFN / (AFN + ATP)   
    Ahter = (Afar + Afrr) / 2

    Vacc = get_acc(all_vidlabels, all_predsV)
    VF1  = get_f1(all_vidlabels, all_predsV)
    Vbacc, Vroc_auc = evaluate_auc(all_vidlabels, all_predsV, all_pos_scoresV)
    VTN, VFP, VFN, VTP, = confusion_matrix(all_vidlabels, all_predsV).ravel()
    Vreal_recall = VTN / (VTN + VFP)
    Vfake_recall = VTP / (VTP + VFN)
    Veer = get_eer(all_vidlabels, all_pos_scoresV)

    Vfar = VFP / (VFP + VTN)
    Vfrr = VFN / (VFN + VTP)
    Vhter = (Vfar + Vfrr) / 2
    result = 'model:{},Total images:{},acc:{:.6f},F1:{:.6f},bACC:{:.6f},RR:{:.6f},FR:{:.6f},ROC_AUC:{:.6f},EER:{:.6f},' \
            'HTER:{:.6f},TN:{},FN:{},TP:{},FP:{},\n,Aacc:{:.6f},AF1:{:.6f},AbACC:{:.6f},ARR:{:.6f},AFR:{:.6f},AROC_AUC:{:.6f},AEER:{:.6f},' \
            'AHTER:{:.6f},ATN:{},AFN:{},ATP:{},AFP:{},\n,Vacc:{:.6f},VF1:{:.6f},VbACC:{:.6f},VRR:{:.6f},VFR:{:.6f},VROC_AUC:{:.6f},VEER:{:.6f},' \
            'VHTER:{:.6f},VTN:{},VFN:{},VTP:{},VFP:{}' \
    .format(str(step), len(all_labels), acc, F1, bacc, real_recall, fake_recall, roc_auc, eer, hter, TN, FN, TP, FP, \
        Aacc, AF1, Abacc, Areal_recall, Afake_recall, Aroc_auc, Aeer, Ahter, ATN, AFN, ATP, AFP,\
        Vacc, VF1, Vbacc, Vreal_recall, Vfake_recall, Vroc_auc, Veer, Vhter, VTN, VFN, VTP, VFP)
    print(step,result)
    #results.append(result + '\n')    
    return result