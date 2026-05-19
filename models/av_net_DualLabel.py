"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from einops import rearrange, repeat


# class CompositionClassifier(nn.Module):

#     def __init__(self, input_dim, num_classes, normalization_sign=False):
#         super().__init__()
       
#         self.mlp = nn.Linear(input_dim*2, input_dim)
#         self.normalization = normalization_sign

#     def forward(self, f1, f2):
#         """
#         :param f1: other modality (e.g. audio or vision)
#         :param f2: video modality
#         :return:
#         """
#         if self.normalization:
#             f1_n = F.normalize(f1, dim=2)
#             f2_n = F.normalize(f2, dim=2)
#             residual = torch.cat((f1_n, f2_n), 2)
#         else:
#             residual = torch.cat((f1, f2), 2)

#         #### compose two modalities by residual learning
#         residual = self.mlp(residual) ### a simple MLP
#         feature = f1 + residual # other modality + residual (default)


#         return feature

class AdapComposition(nn.Module):

    def __init__(self, input_dim,  normalization_sign=True):
        super().__init__()
        # half_input_dim = int(input_dim / 2)
        print("----------comein AdapCOM-----------------------")
        self.mlp = nn.Linear(input_dim * 2, input_dim)
        # self.fc = nn.Linear(input_dim, num_classes)
        self.alpha =nn.Parameter(torch.randn(1))
        self.normalization = normalization_sign

    def forward(self, f1, f2,added_modal):
        """
        ,
        :param f1: audioBatch
        :param f2: videoBatch
        :return:
        """
        if self.normalization:
            # print ("nomrlized")
            f1_n = F.normalize(f1, dim=2)
            f2_n = F.normalize(f2, dim=2)
            residual = torch.cat((f1_n, f2_n), 2)
        else:
            residual = torch.cat((f1, f2), 2)
        # pdb.set_trace()
        #### compose two modalities by residual learning
        residual = self.mlp(residual)  ### a simple MLP
        #if added_modal=="Ada":
        #print ("Ada")factor = torch.gt((torch.sigmoid(self.alpha),0.5)).float()
        factor = torch.gt(torch.sigmoid(self.alpha),0.5).float()
        feature = self.alpha*f1+(1.0-self.alpha)*f2 + residual  # other modality + residual (default)



        return feature

class CompositionClassifier(nn.Module):

    def __init__(self, input_dim, num_classes, normalization_sign=True):
        super().__init__()
        #half_input_dim = int(input_dim / 2)
        print ("----------comein MoDALCOM-----------------------")
        self.mlp = nn.Linear(input_dim*2, input_dim)
        #self.fc = nn.Linear(input_dim, num_classes)
        self.normalization = normalization_sign
        

    def forward(self, f1, f2,added_modal):
        """
        ,
        :param f1: audioBatch
        :param f2: videoBatch
        :return:
        """
        if self.normalization:
            #print ("nomrlized")
            f1_n = F.normalize(f1, dim=2)
            f2_n = F.normalize(f2, dim=2)
            residual = torch.cat((f1_n, f2_n), 2)
        else:
            residual = torch.cat((f1, f2), 2)
        #pdb.set_trace()
        #### compose two modalities by residual learning
        residual = self.mlp(residual)  ### a simple MLP
        if added_modal=="V":
            feature = f2 + residual  # other modality + residual (default)
        elif added_modal=="A":
            feature = f1 + residual
        
        ## perform classification here
        #out = self.fc(feature)

        return feature

def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings#, mask

class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch


class AVNet_Global_Dropout_ModalComps_woFE(nn.Module):

    """
    # An audio-visual speech transcription model based on the Transformer architecture.
    # Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
    #               A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
    #               from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    # Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    # Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
    #              is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    # Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    # Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses,TokenDrop,add_modal='A', pool = 'cls' ):
        super(AVNet_Global_Dropout_ModalComps_woFE, self).__init__()
        
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dModel))
        self.pool = pool
        self.TokenDrop=TokenDrop
        self.add_modal=add_modal
        print("version!!!WO_FakeEncoder AVNet_Global_Dropout_ModalComps",self.TokenDrop,self.add_modal)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        if self.add_modal=="N":
            self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        #if self.add_modal=="Ada":
         #   print ("yes!!")
          #  self.jointConv = AdapComposition(input_dim=512, normalization_sign=True)
        else:
            if self.add_modal == "Ada":
                 print ("yes!!")
                 self.jointConv = AdapComposition(input_dim=512, normalization_sign=True)
            else:
                 print ("no!!")
                 self.jointConv = CompositionClassifier(input_dim=512, num_classes=2, normalization_sign=True)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        #self.TimeDecoder = nn.TransformerEncoder(encoderLayer, num_layers=1)
        self.ClassDecoder = nn.TransformerEncoder(encoderLayer, num_layers=2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dModel),
            nn.Linear(dModel, numClasses)
        )
        
        return


    def forward(self, inputBatch, inputLenBatch):
        #12 exp batchsize; 512/312 notes input feature dimension of v/a
        #a:torch.Size([1132, 12, 321]), v:torch.Size([283, 12, 512])


        audioInputBatch, videoInputBatch = inputBatch
        batchsize = len(inputLenBatch)
        max_length_inBatch= torch.max(inputLenBatch)
        mask = torch.zeros(batchsize,max_length_inBatch,dtype=torch.float)
       
        for e_id, src_len in enumerate(inputLenBatch):
            mask[e_id, src_len:] = 1
        #pdb.set_trace()
        mask = mask.cuda()
        #pdb.set_trace()
        
        if audioInputBatch is not None:
            
            #torch.Size([12, 1132,321])
            audioInputBatch = audioInputBatch.transpose(0, 1)
            #torch.Size([12, 321, 1008])
            if self.training:
                audioInputBatch=drop_tokens(audioInputBatch, self.TokenDrop)
            #torch.Size([12, 512, 283])
            audioBatch = self.audioConv(audioInputBatch.transpose(1, 2))
            #torch.Size([283, 12, 512])
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            #torch.Size([283, 12, 512])
            audioBatch = self.positionalEncoding(audioBatch)
            #torch.Size([283, 12, 512])
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask.bool())
        else:
            audioBatch = None

        if videoInputBatch is not None:
            #torch.Size([ 12,283, 512])
           
            videoInputBatch = videoInputBatch.transpose(0, 1)
            #torch.Size([252, 12, 512])
            if self.training:   
                videoInputBatch = drop_tokens(videoInputBatch, 0.1)
            videoBatch = self.positionalEncoding(videoInputBatch.transpose(0, 1))
            #torch.Size([283, 12, 512])
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask.bool())
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            #torch.Size([283, 12, 512])
            #pdb.set_trace()
            if self.add_modal!="N":

              jointBatch = self.jointConv(audioBatch,videoBatch,self.add_modal)
            else:
              jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
              jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
              #torch.Size([12, 512, 283])
              jointBatch = self.jointConv(jointBatch)
              #torch.Size([283, 12, 512])

              jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()
        #torch.Size([283, 12, 512])
        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask.bool())
        n, b, _ = jointBatch.shape  # batch,num_patches,channels  #

        mask_tokens = torch.cat((torch.zeros(batchsize,1).cuda(),mask),dim=1)
        cls_tokens = repeat(self.cls_token, 'n () d -> n b d', b=b)#turn to multilabelchange
        jointBatch = torch.cat((cls_tokens, jointBatch), dim=0)
        jointBatch = self.ClassDecoder(jointBatch, src_key_padding_mask=mask_tokens.bool())

        classjointBatch= jointBatch.mean(dim=1) if self.pool == 'mean' else jointBatch[0, :]
        fakeNumjointBatch=jointBatch[1, :]
       
        #Batch,classes
        outputBatch = self.mlp_head(classjointBatch)
       
        return outputBatch,classjointBatch#None
#

class AVNet_Global_Dropout_ModalComps(nn.Module):

    """
    # An audio-visual speech transcription model based on the Transformer architecture.
    # Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
    #               A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
    #               from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    # Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    # Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
    #              is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    # Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    # Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses,TokenDrop,add_modal='N', pool = 'cls' ):
        super(AVNet_Global_Dropout_ModalComps, self).__init__()
        
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.fake_token = nn.Parameter(torch.randn(1, 1, dModel))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dModel))
        self.pool = pool
        self.TokenDrop=TokenDrop
        self.add_modal=add_modal
        print("version!!!AVNet_Global_Dropout_ModalComps",self.TokenDrop,self.add_modal)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        if self.add_modal=="N":
            self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        #else:
        #    self.jointConv = CompositionClassifier(input_dim=512, num_classes=2, normalization_sign=True)
        else:
            if self.add_modal == "Ada":
                 
                 self.jointConv = AdapComposition(input_dim=512, normalization_sign=True)
            else:
                 
                 self.jointConv = CompositionClassifier(input_dim=512, num_classes=2, normalization_sign=True)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.TimeDecoder = nn.TransformerEncoder(encoderLayer, num_layers=1)
        self.ClassDecoder = nn.TransformerEncoder(encoderLayer, num_layers=2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dModel),
            nn.Linear(dModel, numClasses)
        )
        self.mlp_FakeNum = nn.Sequential(
            nn.LayerNorm(dModel),
            nn.Linear(dModel, 3)
        )
        return


    def forward(self, inputBatch, inputLenBatch):
        #12 exp batchsize; 512/312 notes input feature dimension of v/a
        #a:torch.Size([1132, 12, 321]), v:torch.Size([283, 12, 512])


        audioInputBatch, videoInputBatch = inputBatch
        batchsize = len(inputLenBatch)
        max_length_inBatch= torch.max(inputLenBatch)
        mask = torch.zeros(batchsize,max_length_inBatch,dtype=torch.float)
       
        for e_id, src_len in enumerate(inputLenBatch):
            mask[e_id, src_len:] = 1
        #pdb.set_trace()
        mask = mask.cuda()
        #pdb.set_trace()
        
        if audioInputBatch is not None:
            
            #torch.Size([12, 1132,321])
            audioInputBatch = audioInputBatch.transpose(0, 1)
            #torch.Size([12, 321, 1008])
            if self.training:
                audioInputBatch=drop_tokens(audioInputBatch, self.TokenDrop)
            #torch.Size([12, 512, 283])
            audioBatch = self.audioConv(audioInputBatch.transpose(1, 2))
            #torch.Size([283, 12, 512])
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            #torch.Size([283, 12, 512])
            audioBatch = self.positionalEncoding(audioBatch)
            #torch.Size([283, 12, 512])
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask.bool())
        else:
            audioBatch = None

        if videoInputBatch is not None:
            #torch.Size([ 12,283, 512])
           
            videoInputBatch = videoInputBatch.transpose(0, 1)
            #torch.Size([252, 12, 512])
            if self.training:   
                videoInputBatch = drop_tokens(videoInputBatch, 0.1)
            videoBatch = self.positionalEncoding(videoInputBatch.transpose(0, 1))
            #torch.Size([283, 12, 512])
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask.bool())
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            #torch.Size([283, 12, 512])
            #pdb.set_trace()
            if self.add_modal!="N":

              jointBatch = self.jointConv(audioBatch,videoBatch,self.add_modal)
            else:
              jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
              jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
              #torch.Size([12, 512, 283])
              jointBatch = self.jointConv(jointBatch)
              #torch.Size([283, 12, 512])

              jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()
        #torch.Size([283, 12, 512])
        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask.bool())
        n, b, _ = jointBatch.shape  # batch,num_patches,channels  #

        mask_tokens = torch.cat((torch.zeros(batchsize,1).cuda(),mask),dim=1)
        fakeNum_tokens = repeat(self.fake_token, 'n () d -> n b d', b=b)#
        jointBatch = torch.cat((fakeNum_tokens, jointBatch), dim=0)
        jointBatch = self.TimeDecoder(jointBatch, src_key_padding_mask=mask_tokens.bool())
        
        mask_tokens = torch.cat((torch.zeros(batchsize,1).cuda(),mask_tokens),dim=1)
        cls_tokens = repeat(self.cls_token, 'n () d -> n b d', b=b)#turn to multilabelchange
        jointBatch = torch.cat((cls_tokens, jointBatch), dim=0)
        jointBatch = self.ClassDecoder(jointBatch, src_key_padding_mask=mask_tokens.bool())

        classjointBatch= jointBatch.mean(dim=1) if self.pool == 'mean' else jointBatch[0, :]
        fakeNumjointBatch=jointBatch[1, :]
       
        #Batch,classes
        outputBatch = self.mlp_head(classjointBatch)
        fakeNum_output=self.mlp_FakeNum(fakeNumjointBatch)
        #return outputBatch,classjointBatch
        return outputBatch,fakeNum_output



class  AVNet_raw(nn.Module):

    """
    # An audio-visual speech transcription model based on the Transformer architecture.
    # Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
    #               A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
    #               from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    # Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    # Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
    #              is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    # Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    # Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses,TokenDrop,add_modal, pool = 'cls' ):
        super(AVNet_raw, self).__init__()
        
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
       
        self.pool = pool
        self.TokenDrop=TokenDrop
        self.add_modal=add_modal
        print("version!!!WO DualLabel Classifier",self.TokenDrop,self.add_modal)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        if self.add_modal=="N":
            self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        else:
            self.jointConv = CompositionClassifier(input_dim=512, num_classes=2, normalization_sign=True)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        
        self.output_framewise = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
 
        return


    def forward(self, inputBatch, inputLenBatch):
        #12 exp batchsize; 512/312 notes input feature dimension of v/a
        #a:torch.Size([1132, 12, 321]), v:torch.Size([283, 12, 512])


        audioInputBatch, videoInputBatch = inputBatch
        batchsize = len(inputLenBatch)
        max_length_inBatch= torch.max(inputLenBatch)
        mask = torch.zeros(batchsize,max_length_inBatch,dtype=torch.float)
       
        for e_id, src_len in enumerate(inputLenBatch):
            mask[e_id, src_len:] = 1
        #pdb.set_trace()
        mask = mask.cuda()
        #pdb.set_trace()
        
        if audioInputBatch is not None:
            
            #torch.Size([12, 1132,321])
            audioInputBatch = audioInputBatch.transpose(0, 1)
            #torch.Size([12, 321, 1008])
            if self.training:
                audioInputBatch=drop_tokens(audioInputBatch, self.TokenDrop)
            #torch.Size([12, 512, 283])
            audioBatch = self.audioConv(audioInputBatch.transpose(1, 2))
            #torch.Size([283, 12, 512])
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            #torch.Size([283, 12, 512])
            audioBatch = self.positionalEncoding(audioBatch)
            #torch.Size([283, 12, 512])
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask.bool())
        else:
            audioBatch = None

        if videoInputBatch is not None:
            #torch.Size([ 12,283, 512])
           
            videoInputBatch = videoInputBatch.transpose(0, 1)
            #torch.Size([252, 12, 512])
            if self.training:   
                videoInputBatch = drop_tokens(videoInputBatch, 0.1)
            videoBatch = self.positionalEncoding(videoInputBatch.transpose(0, 1))
            #torch.Size([283, 12, 512])
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask.bool())
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            #torch.Size([283, 12, 512])
            #pdb.set_trace()
            if self.add_modal!="N":

              jointBatch = self.jointConv(audioBatch,videoBatch,self.add_modal)
            else:
              jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
              jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
              #torch.Size([12, 512, 283])
              jointBatch = self.jointConv(jointBatch)
              #torch.Size([283, 12, 512])

              jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()
        #torch.Size([283, 12, 512])
        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask.bool())
        #torch.Size([12, 512, 283])
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        #torch.Size([12, 2, 283])
        jointBatch = self.output_framewise(jointBatch)
        ##torch.Size([12, 2, 283])
        
        jointBatch=torch.mul((1-mask),jointBatch.transpose(0,1)).transpose(0,1)
        jointBatch=torch.sum(jointBatch,2).transpose(0,1)/inputLenBatch
        #pdb.set_trace()
        #torch.Size([283, 12, 2])
        outputBatch = jointBatch.transpose(0, 1)
        #Batch,classes
        
       
        return outputBatch, None




class  AVNet_woTimeDecoder(nn.Module):

    """
    # An audio-visual speech transcription model based on the Transformer architecture.
    # Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
    #               A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
    #               from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    # Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    # Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
    #              is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    # Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    # Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses,TokenDrop,add_modal, pool = 'cls' ):
        super(AVNet_woTimeDecoder, self).__init__()
        
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
       
        self.pool = pool
        self.TokenDrop=TokenDrop
        self.add_modal=add_modal
        self.fake_token = nn.Parameter(torch.randn(1, 1, dModel))
        print("version!!!WO AVNet_woTimeDecoder",self.TokenDrop,self.add_modal)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        if self.add_modal=="N":
            self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        else:
            self.jointConv = CompositionClassifier(input_dim=512, num_classes=2, normalization_sign=True)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.FakeDecoder = nn.TransformerEncoder(encoderLayer, num_layers=1)
        self.mlp_FakeNum = nn.Sequential(
            nn.LayerNorm(dModel),
            nn.Linear(dModel, 3)
        )
        self.output_framewise = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)

        return


    def forward(self, inputBatch, inputLenBatch):
        #12 exp batchsize; 512/312 notes input feature dimension of v/a
        #a:torch.Size([1132, 12, 321]), v:torch.Size([283, 12, 512])


        audioInputBatch, videoInputBatch = inputBatch
        batchsize = len(inputLenBatch)
        max_length_inBatch= torch.max(inputLenBatch)
        mask = torch.zeros(batchsize,max_length_inBatch,dtype=torch.float)
       
        for e_id, src_len in enumerate(inputLenBatch):
            mask[e_id, src_len:] = 1
        #pdb.set_trace()
        mask = mask.cuda()
        #pdb.set_trace()
        
        if audioInputBatch is not None:
            
            #torch.Size([12, 1132,321])
            audioInputBatch = audioInputBatch.transpose(0, 1)
            #torch.Size([12, 321, 1008])
            if self.training:
                audioInputBatch=drop_tokens(audioInputBatch, self.TokenDrop)
            #torch.Size([12, 512, 283])
            audioBatch = self.audioConv(audioInputBatch.transpose(1, 2))
            #torch.Size([283, 12, 512])
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            #torch.Size([283, 12, 512])
            audioBatch = self.positionalEncoding(audioBatch)
            #torch.Size([283, 12, 512])
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask.bool())
        else:
            audioBatch = None

        if videoInputBatch is not None:
            #torch.Size([ 12,283, 512])
           
            videoInputBatch = videoInputBatch.transpose(0, 1)
            #torch.Size([252, 12, 512])
            if self.training:   
                videoInputBatch = drop_tokens(videoInputBatch, 0.1)
            videoBatch = self.positionalEncoding(videoInputBatch.transpose(0, 1))
            #torch.Size([283, 12, 512])
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask.bool())
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            #torch.Size([283, 12, 512])
            #pdb.set_trace()
            if self.add_modal!="N":

              jointBatch = self.jointConv(audioBatch,videoBatch,self.add_modal)
            else:
              jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
              jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
              #torch.Size([12, 512, 283])
              jointBatch = self.jointConv(jointBatch)
              #torch.Size([283, 12, 512])

              jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()
        #torch.Size([283, 12, 512])
        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask.bool())
        n, b, _ = jointBatch.shape  # batch,num_patches,channels  #

        mask_tokens = torch.cat((torch.zeros(batchsize,1).cuda(),mask),dim=1)
        fakeNum_tokens = repeat(self.fake_token, 'n () d -> n b d', b=b)#
        jointBatch = torch.cat((fakeNum_tokens, jointBatch), dim=0)
        jointBatch = self.FakeDecoder(jointBatch, src_key_padding_mask=mask_tokens.bool())
        #pdb.set_trace()
        fakeNumjointBatch=jointBatch[0, :]
        TimejointBatch=jointBatch[1:, :]
        fakeNum_output=self.mlp_FakeNum(fakeNumjointBatch)


        #torch.Size([12, 2, 283])
        jointBatch = self.output_framewise(TimejointBatch.transpose(0, 1).transpose(1, 2))
        ##torch.Size([12, 2, 283])
        jointBatch=torch.mul((1-mask),jointBatch.transpose(0,1)).transpose(0,1)
        jointBatch=torch.sum(jointBatch,2).transpose(0,1)/inputLenBatch
        #pdb.set_trace()
        #torch.Size([283, 12, 2])
        outputBatch = jointBatch.transpose(0, 1)
        #Batch,classes
        
        return outputBatch, torch.mean(TimejointBatch,dim=0)
        #return outputBatch, fakeNum_output

