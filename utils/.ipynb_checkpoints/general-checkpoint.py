"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_cer, compute_wer, evaluate_auc, confusion_matrix, get_acc, get_eer
import pdb
import torch.nn as nn
#from .decoders import ctc_greedy_decode, ctc_search_decode


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad==True])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams, global_label):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    
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
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                pass
            
            optimizer.zero_grad()
            model.train()
            ## outputBatch torch.Size([283, 12, 2])->torch.Size([12, 2, 283])
            #pdb.set_trace()
            outputBatch = model(inputBatch, inputLenBatch)#.transpose(1,2).transpose(0,2)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, labelBatch)
            loss.backward()
            optimizer.step()

            trainingLoss = trainingLoss + loss.item()
    trainingLoss = trainingLoss/len(trainLoader)  
    return trainingLoss

def evaluate(model, evalLoader, step, device, global_label):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """
    all_preds = []
    all_labels = []
    all_pos_scores = []
    results = []
    if global_label==False:
        for batch, (inputBatch, inputLenBatch, labelBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                            ncols=75)):
        
            inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device))
            #torch.Size([252, 12, 1])
            inputLenBatch = (inputLenBatch.int()).to(device)#.transpose(0,1)#, (targetLenBatch.int()).to(device)
            labelBatch = (labelBatch.long()).to(device)#.transpose(0,1)
            # opmode = np.random.choice(["AO", "VO", "AV"],
            #                          p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
            # if opmode == "AO":
            #     inputBatch = (inputBatch[0], None)
            # elif opmode == "VO":
            #     inputBatch = (None, inputBatch[1])
            # else:
            #     pass

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
            # opmode = np.random.choice(["AO", "VO", "AV"],
            #                          p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
            # if opmode == "AO":
            #     inputBatch = (inputBatch[0], None)
            # elif opmode == "VO":
            #     inputBatch = (None, inputBatch[1])
            # else:
            #     pass

            model.eval()
            with torch.no_grad():
                outputBatch = model(inputBatch,inputLenBatch)#.transpose(1,2).transpose(0,2)
           
            #pdb.set_trace()
            #post_function = nn.Softmax(dim=1)
            #outputBatch = post_function(outputBatch)
            _, predicted = torch.max(outputBatch , 1)
            #pdb.set_trace()
            outputs =outputBatch[:, -1].view(-1)
            all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labelBatch.cpu().numpy().tolist())
    #pdb.set_trace()
    
    acc = get_acc(all_labels, all_preds)
    bacc, roc_auc = evaluate_auc(all_labels, all_preds, all_pos_scores)
    TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
    real_recall = TN / (TN + FP)
    fake_recall = TP / (TP + FN)
    eer = get_eer(all_labels, all_pos_scores)

    far = FP / (FP + TN)
    frr = FN / (FN + TP)
    hter = (far + frr) / 2
    result = 'model:{},Total images:{},acc:{:.6f},bACC:{:.6f},RR:{:.6f},FR:{:.6f},ROC_AUC:{:.6f},EER:{:.6f},' \
             'HTER:{:.6f},TN:{},FN:{},TP:{},FP:{}' \
        .format(str(step), len(all_labels), acc, bacc, real_recall, fake_recall, roc_auc, eer, hter, TN, FN, TP, FP)
    #print(step,result)
    results.append(result + '\n')    
    return result
