"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax



def prepare_main_input(audioFile, visualFeaturesFile, reqInpLen, audioParams, videoParams):

    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

   

    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    if inputAudio is None or len(inputAudio)==0 or np.abs(inputAudio).any()==0 or np.isnan(inputAudio).any() or np.isinf(inputAudio).any():
          raise ValueError(f"Empty audio data at index {audioFile}")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

   
    
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))
    print("inputAudio max:", np.max(np.abs(inputAudio)))
    print("inputAudio mean/std:", np.mean(inputAudio), np.std(inputAudio))

    #computing STFT and taking only the magnitude of it
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    #loading the visual features
   # vidInp = np.load(visualFeaturesFile)
    # here (560, 321) (123, 512) almost 4:1
    #print(audInp.shape,vidInp.shape)
    if visualFeaturesFile is not None:
        vidInp = np.load(visualFeaturesFile)
    else:
        dummy_len = int(np.ceil(len(audInp) / 4))
        vidInp = np.zeros((dummy_len, 512), dtype=np.float32)  # 假设视觉特征维度为 512

    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    #print(audioInp.shape,vidInp.shape)
    inp = (audInp,vidInp)
    
    inpLen = torch.tensor(inpLen)
    #torch.Size([992, 321]) torch.Size([248, 512]) tensor(248)

    print("shape",audInp.shape,vidInp.shape,inpLen)
   

    return inp, inpLen


def collate_fn_global_label(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    

    inputLenBatch = torch.stack([data[1] for data in dataBatch])
    labelBatch = torch.stack([data[2] for data in dataBatch])

    return inputBatch, inputLenBatch,labelBatch


def collate_fn_global_labelDrop(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    

    inputLenBatch = torch.stack([data[1] for data in dataBatch])
    labelBatch = torch.stack([data[2] for data in dataBatch])
    ModalBatch = ([data[3] for data in dataBatch])

    return inputBatch, inputLenBatch,labelBatch,ModalBatch


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    data 是一个三元组: ((modality1, modality2), inputLen, label)
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    

    inputLenBatch = torch.stack([data[1] for data in dataBatch])
    labelBatch = pad_sequence([data[2] for data in dataBatch],padding_value=-1)

    return inputBatch, inputLenBatch,labelBatch

def collate_fn_test(dataBatch):
    """
    Collate function definition used in Dataloaders.
    data list order
    inputBatch, targetBatch, inputLenBatch, targetLenBatch
    """
    # padding to max sequence in Batch
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    # if not any(data[1] is None for data in dataBatch):
    #     targetBatch = torch.cat([data[1] for data in dataBatch])
    # else:
    #     targetBatch = None
    
    inputLenBatch = pad_sequence([data[1] for data in dataBatch])
    labelBatch = torch.stack([data[2] for data in dataBatch])
    # if not any(data[3] is None for data in dataBatch):
    #     targetLenBatch = torch.stack([data[3] for data in dataBatch])
    # else:
    #     targetLenBatch = None
    #print ("in collate fn",inputBatch.shape, inputLenBatch)
    #print ("collate",len(dataBatch),type(dataBatch))
    return inputBatch, inputLenBatch,labelBatch