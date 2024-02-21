import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer
from torch.nn.parallel import DataParallel
from transformers import BlipModel

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

    def forward(self, vid):
        video_features = self.blip.get_image_features(vid)
        return video_features
    
class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        self.device_ids = [2, 4, 5, 6, 7]
        self.blip = Model()
        
        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        blip = DataParallel(model, device_ids=self.device_ids)
    
    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        
        # text_data = text_data.to(torch.device('cuda:2'))
        
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        # video_data = video_data.to(torch.device('cuda:2'))
        
        model = self.blip.to(torch.device('cuda:2'))
        
        video_features = blip(video_data)
        
        # print(len(text_data['input_ids']), video_data.shape)
        
        text_features = self.blip.blip.get_text_features(**text_data)
        # print(video_features.shape, text_features.shape)
        
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
