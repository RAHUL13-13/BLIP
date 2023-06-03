import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from transformers import BlipModel, AutoProcessor
from torch.nn.parallel import DataParallel
import cv2
import random
import numpy as np
import torch

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def load_frames_from_video(video_path, num_frames, sample='uniform'):
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                print(processor(images=[frame,frame], return_tensors="pt"))
                # print(model.get_image_features(**processor(images=frame, return_tensors="pt")))
                break
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs


load_frames_from_video('/shared/home/v_rahul_pratap_singh/local_scratch/videoRetrieval/xpool/data/MSVD/YouTubeClips/_1vy2HIN60A_32_40.avi', \
    12)

image = cv2.imread('/shared/home/v_rahul_pratap_singh/local_scratch/videoRetrieval/xpool_base/xpool/output.png')
# print(image)
# image = torch.from_numpy(image)
print(model.get_image_features(**processor(images=image, return_tensors="pt")))

exit()

# Load the model
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
model.to("cuda:0")

device_ids = [0,1,2,7]
model = DataParallel(model, device_ids=device_ids)
# Load the processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the images
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image2 = Image.open(requests.get(url, stream=True).raw)
image3 = Image.open(requests.get(url, stream=True).raw)
image4 = Image.open(requests.get(url, stream=True).raw)
image5 = Image.open(requests.get(url, stream=True).raw)

images = [image, image2, image3, image4, image5]*24

# text_features = model.module.get_text_features(**text_data)
# video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
video_data = processor(images=images, return_tensors="pt")['pixel_values']
print(video_data.shape)
batch_size = len(video_data)
device_ids = [0, 1, 2, 7]
split_size = (batch_size + len(device_ids) - 1) // len(device_ids)
split_video_data = [video_data[i * split_size: (i + 1) * split_size] for i in range(len(device_ids))]

# Pass the input data through the model
output = []
for i, data in enumerate(split_video_data):
    with torch.cuda.device(device_ids[i]):
        print(data.shape)
        output.append(model.module(data))

# Concatenate the outputs from different GPUs
output = torch.cat(output)
exit()
# for i in range(0, batch_size, 2):
# # Create the original tensor
# original_tensor = torch.tensor([[[1, 2], [3, 4]]])

# # Repeat the tensor n times along dimension 0
# n = 3
# repeated_tensor = original_tensor.repeat(n,1,1)

# # Print the repeated tensor
# print(repeated_tensor)

# exit()

# matrix1 = torch.randint(1,4,(2, 2, 3,4))
# matrix2 = torch.randint(1,4,(2, 2, 3,4))
# a = matrix1
# b = matrix2
# print(a.permute(1,2,3,0))
# print(b.permute(1,2,3,0))
# num_vids, _, max_text_per_vid, embed_dim = matrix1.shape

# matrix1 = matrix1.view(num_vids*max_text_per_vid*num_vids, embed_dim)

# matrix2 = matrix2.view(num_vids*max_text_per_vid*num_vids, embed_dim)

# sims = torch.bmm(matrix2.unsqueeze(1), matrix1.unsqueeze(-1)).squeeze()
# print(sims)
# sims = sims.reshape(num_vids, max_text_per_vid*num_vids).t().view(num_vids, max_text_per_vid, num_vids)
# print(sims)


# # num_vids, _, max_text_per_vid, embed_dim = a.shape

# # a = a.permute(1,2,0,3)
# # # vid text embed vid
# # a = a.reshape(num_vids*max_text_per_vid*num_vids, embed_dim)

# # b = b.permute(1,2,0,3)
# # # vid text vid embed
# # b = b.reshape(num_vids*max_text_per_vid*num_vids, embed_dim)

# # sims1 = torch.bmm(b.unsqueeze(1), a.unsqueeze(-1)).squeeze()
# # print(sims1.shape)
# # sims1 = sims1.reshape(num_vids, max_text_per_vid, num_vids)

# # print(sims==sims1)
# exit()

# # Create two matrices
# matrix1 = torch.randint(1,4,(2, 2, 3))
# matrix2 = torch.randint(1,4,(2, 2, 3))

# # Reshape the matrices to size (a*b*c, 512)
# matrix1_reshaped = matrix1.view(4, 3)
# matrix2_reshaped = matrix2.view(4, 3)

# # Compute the dot product along the last dimension
# dot_product = torch.bmm(matrix1_reshaped.unsqueeze(1), matrix2_reshaped.unsqueeze(-1)).squeeze()

# # Reshape the output tensor to size (a, b, c)
# dot_product = dot_product.view(2, 2)

# print(matrix1,"\n",matrix2, "\n",matrix1_reshaped, "\n",dot_product)
# exit()

# a = torch.randn(2, 2, 3, 4)
# permuted_a = a.permute(1, 2, 3, 0)
# view_a = permuted_a.view(2*3, 4, 2)

# print("Original tensor a:")
# print(a)
# print("Shape of a:", a.shape)

# print("Permutated tensor permuted_a:")
# print(permuted_a)
# print("Shape of permuted_a:", permuted_a.shape)

# print("after view tensor view_a:")
# print(view_a)
# print("Shape of view_a:", view_a.shape)

# exit()

# # Create a 3x4x3 tensor
# t = torch.randn(3, 4, 3).permute(1,0,2)

# # Extract the diagonal elements along the second and third dimensions
# diagonal = torch.diagonal(t, dim1=1, dim2=2)

# print("Original tensor:\n", t)
# print("Diagonal elements:\n", diagonal)

# exit()

# b = torch.randint(1,4, (4, 1, 3))
# print(b)

c = torch.randint(1,4, (4, 3, 2))
print(c)

res = torch.bmm(b, c)
print(res)

exit()

b = torch.randn(2, 3, 4)
print(b)

b = b.unsqueeze(2)
print(b)

b = b.view(2*3, 1, 4)
print(b)

exit()



import torch
from transformers import CLIPConfig, CLIPModel

# Load the CLIP-ViT-B/32 model configuration
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# Create a CLIPModel instance
model = CLIPModel(config)

# Print the total number of parameters in the ViT portion of the model
vit_params = sum(p.numel() for p in model.visual.transformer.parameters())
print("Total parameters in ViT portion: ", vit_params)
