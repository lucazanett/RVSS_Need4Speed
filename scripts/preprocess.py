import torch
import torchvision.transforms as transforms


def preprocess_image(im,final_height = 40, final_width = 60):

    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((final_height, final_width)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

