"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop
from sklearn import metrics
# from .data.transforms import NormalizeVideo, ToTensorVideo
# from .data.dataset_clips import ForensicsClips, CelebDFClips, DFDCClips
# from .data.samplers import ConsecutiveClipSampler
#from .transforms import NormalizeVideo, ToTensorVideo
#from .samplers import ConsecutiveClipSampler
from scipy.io import wavfile
import numpy as np
import os 
import torch
import pdb
import bisect
from torch.utils.data import DataLoader

# transform = Compose(
#         [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
#     )
# class Cross_Data(Dataset):

#     """
#     A custom dataset class for the LRS2 main (includes train, val, test) dataset
#     """

#     def __init__(self, mode, datadir, reqInpLen, charToIx, stepSize, global_label, audioParams, videoParams):
#         super(Cross_Data, self).__init__()
#         self.mode=mode       
#         filesList = list()
#         #load FAKEAVCELEB
#         if self.mode=="train":
#            datadir=datadir.format("TRAIN")
#         elif self.mode=="test":
#            datadir=datadir.format("TEST")
        
#         fdatadir=os.path.join(datadir,"fake")
#         real_filesList=[]
#         fake_filesList=[]
#         print ("____data path_____",datadir)
#         visualAC_type=["REAL_A","FAKE_FACESWAP_C","FAKE_FSGAN_C","FAKE_W2L_C" ]
#         for type in visualAC_type:
#             datatypedir=os.path.join (datadir,type)
#             for root, dirs, files in os.walk(datatypedir):
#                 for file in files:
#                     if file.endswith(".npy"):
#                         filesList.append(os.path.join(root, file[:-4]))
#         self.adatalist = filesList 
#         #print ("all wav file",len(self.datalist))
#         self.reqInpLen = reqInpLen
#         #self.charToIx = charToIx
#         self.mode = mode
#         self.stepSize = stepSize
#         self.global_label = global_label
#         self.audioParams = audioParams
#         self.videoParams = videoParams
#         # _, self.noise = wavfile.read(noiseParams["noiseFile"])
#         # self.noiseSNR = noiseParams["noiseSNR"]
#         # self.noiseProb = noiseParams["noiseProb"]
#         return

#     def __getitem__(self, index):
#         #using the same procedure as in pretrain dataset class only for the train dataset
#         audioFile = self.adatalist[index] + "_16khz.wav"
#         visualFeaturesFile = self.adatalist[index] + ".npy"
        
#         #print ("999",audioFile,visualFeaturesFile,"999")
#         #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        
#         inp, inpLen= prepare_main_input(audioFile, visualFeaturesFile, self.reqInpLen, self.audioParams, self.videoParams)
#         if 'REAL_A' in self.adatalist[index] :
#             label = torch.tensor(0)#.repeat(inpLen)

#         else:
#             label = torch.tensor(1)#.repeat(inpLen)
       
#         if self.mode == "train" and self.global_label==False:
#           label= label.repeat(inpLen)
#         elif self.mode =="test" and self.global_label==False:
#           inpLen = torch.ones(inpLen,1)
#         return inp, inpLen, label###


#     def __len__(self):
#         #using step size only for train dataset and not for val and test datasets because
#         #the size of val and test datasets is smaller than step size and we generally want to validate and test
#         #on the complete dataset
#         #if self.dataset == "train":
#          #   return self.stepSize
#         #else:
#             return len(self.adatalist)


class DFDC_LipForensics(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, mode, datadir,frames_per_clip, grayscale=False, transform=None,):
        super(DFDC_LipForensics, self).__init__()
        self.mode=mode   
        self.frames_per_clip = frames_per_clip
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []    
        self.filesList = list()
        self.frames=[]
        #### load DFDC_AVSR
        real_datadir=os.path.join(datadir,"real")
        fake_datadir=os.path.join(datadir,"fake")
        real_filesList=[]
        fake_filesList=[]
        print ("____data path_____",datadir)
        for root, dirs, files in os.walk(real_datadir):
            for file in files:
                if file.endswith("0.png"):
                    if not os.path.exists(os.path.join(root, file[:-5]+"1.png")):
                        real_filesList.append(os.path.join(root, file[:-4])+".png")
        for root, dirs, files in os.walk(fake_datadir):
            for file in files:
                if file.endswith("0.png"):
                    if not os.path.exists(os.path.join(root, file[:-5]+"1.png")):
                        fake_filesList.append(os.path.join(root, file[:-4]+".png"))
        #pdb.set_trace()
        if self.mode=="train":
            self.filesList=real_filesList[:-1300]+fake_filesList[:-1300]
            #self.filesList=real_filesList[:130]+fake_filesList[:130]
        elif self.mode=="test":
            self.filesList=real_filesList[-1300:]+fake_filesList[-1300:]
            #self.filesList=real_filesList[-13:]+fake_filesList[-13:]

        print ("total data ", len(self.filesList))#train 14954,test2600
        for video_frames in self.filesList:
            with Image.open(video_frames) as pil_img:
                # if self.grayscale:
                #     pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            #num_frames = min(img.shape[1]//112,250)
            num_frames =min(img.shape[1]//112,100)
            num_clips = num_frames // frames_per_clip
            self.frames.append(img)
            self.clips_per_video.append(num_clips)
        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        print ("clips_num",self.cumulative_sizes[-1])
        print(len(self.filesList),len(self.frames))
    
    def __len__(self):
        
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.filesList[video_idx]
        
        frames = self.frames[video_idx]
        img=np.split(frames,frames.shape[1]//112,axis=1)

       

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip
        #print ("======",start_idx,end_idx,len(img))
        #pdb.set_trace()
        sample = []
        for fidx in range(start_idx, end_idx, 1):
            #print(img[0].shape)
            if fidx>=len(img):print ("===========",video_idx,clip_idx,len(img))
            sample.append(img[fidx])

        sample = np.stack(sample)

        return sample, video_idx, path

    def __getitem__(self, idx):
        #using the same procedure as in pretrain dataset class only for the train dataset
        sample, video_idx, video_path = self.get_clip(idx)      
        
        if 'real' in  video_path:
            label = 0

        else:
            label = 1
        label = torch.tensor(label, dtype=torch.float32)

        #label = torch.from_numpy(np.array(label))
        #sample = torch.from_numpy(sample).unsqueeze(-1)
        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx


# from config import args
# gpuAvailable = torch.cuda.is_available()
# data=DFDC_LipForensics("test", args["DATA_DIRECTORY"], frames_per_clip=25, grayscale=False,transform=transform)
# kwargs = {"num_workers":args["NUM_WORKERS"], "pin_memory":True} if gpuAvailable else {}
# pdb.set_trace()
# testLoader = DataLoader(data, batch_size=args["BATCH_SIZE"],shuffle=True, **kwargs)    