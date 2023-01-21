import cv2
import os
from PIL import Image
import torchvision.transforms as T
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Transform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 

def get_views(img_path, save_path = "media",image_size=None):
    # 获取图像增强策略
    save_path = os.path.join(os.getcwd(), "media")
    aug = Transform(image_size=image_size)
    # 读取图片
    image_path = img_path
    img = Image.open(image_path)
    # 进行图像增强，产生两个视图
    v1,v2 = aug(img)
    # 保存图片
    cv2.imwrite(os.path.join(save_path,"view1.png"), v1.permute(1, 2, 0).numpy()*255)
    cv2.imwrite(os.path.join(save_path,"view2.png"), v2.permute(1, 2, 0).numpy()*255)

    return v1, v2