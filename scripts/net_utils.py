

from torchvision import transforms


from preprocess import PreProcessImage

def get_transform():
    transform = transforms.Compose([
        PreProcessImage(),
        transforms.ToPILImage(),

        transforms.Resize((40, 60)),
        transforms.ColorJitter(brightness=0.15, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    return transform
