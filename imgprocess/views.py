import os
import sys
import torch

from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse
from .models import Image

from deepM.augmentation import get_views
from deepM.utils import get_backbone, get_level_feature_map, get_cam

BASE_PATH = os.getcwd()
model = None
model_name = ''
# Create your views here.
def index2(request):
    """主页面
    """
    global model, model_name
    model = None
    model_name = ''
    content = {"img_path":'default.png',
                "key": 0}
    return render(request, "index2.html", content)

def pic_handle(request):
    """上传并保存图片
    """
    global model, model_name
    model = None
    model_name = ''

    f1=request.FILES.get('pic')

    if f1 is None:
        content = {"img_path":'default.png',
                    "do_success": 2,
                    "key": 0}
        return render(request, "index2.html", content)
    
    input_img = "input_img.png"
    fname_media='./media/input_img.png'

    with open(fname_media,'wb+') as pic:    
        for c in f1.chunks():
            pic.write(c)

    # request.session["msg"] = " 上传成功！"

    ctx = {
        'img_path': input_img,
        'key': 0,
    }
    return render(request, 'index2.html', ctx)

def load_resnet_18(request):
    """加载预训练的resnet18网络
    """
    global model, model_name
    try:
        model = get_backbone("resnet18_ml_backbone",castrate=False)
        model_name = 'resnet_18'
        eval_from = './deepM/mlrl-fusion-tiny_imagenet-ep200-200-resnet18_ml_backbone.pth'
        save_dict = torch.load(eval_from, map_location='cpu')
        msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    except:
        model = None
        model_name == ''

    do_success = 3
    if model is None:
        do_success = 0 # 如果模型未加载成功，do_success=0
    
    ctx = {
        'img_path': 'input_img.png',
        'do_success': do_success,
        'key': 0,
    }

    return render(request, "index2.html", ctx)

def load_resnet_50(request):
    """加载预训练的resnet50网络
    """
    ...

def processing(request):
    """对图像进行处理，并保存结果
    """
    # 对图片进行处理：模型预测，中间图像保存
    global model
    if model is None:
        ctx = {
            'img_path': 'input_img.png',
            'do_success': 0,
            'key': 0,
        }
        return render(request, "index2.html", ctx)
    input_imgp = './media/input_img.png'
    try:
        v1, v2 = get_views(input_imgp,image_size=448)
        # model = get_backbone("resnet18_ml_backbone",castrate=False)
        # eval_from = './deepM/mlrl-fusion-tiny_imagenet-ep200-200-resnet18_ml_backbone.pth'
        # save_dict = torch.load(eval_from, map_location='cpu')
        # msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
        x_v1, low_v1, mid_v1, high_v1 = model.forward(v1.view((1,3,448,448)))
        x_v2, low_v2, mid_v2, high_v2 = model.forward(v2.view((1,3,448,448)))
        get_level_feature_map(low_v1, low_v2, img_prefix="low", t=2, gap=4, num_p=3)
        get_level_feature_map(mid_v1, mid_v2, img_prefix="mid", t=10, gap=3, num_p=6)
        get_level_feature_map(high_v1, high_v2, img_prefix="high", t=15, gap=1, num_p=12)

        get_cam(img_path=input_imgp)

        ctx = {
            'img_path': 'input_img.png',
            'do_success': 1,
            'key': 0,
        }
    except:
        ctx = {
            'img_path': 'input_img.png',
            'do_success': 0,
            'key': 0,
        }
    return render(request, 'index2.html', ctx)

def show_classification_result(request):
    """显示分类结果
    """
    ...

def show_enhanced_views(request):
    """显示增强后视图"""

    global model
    if model is None:
        ctx = {
            'img_path': 'input_img.png',
            'do_success': 0,
            'key': 0,
        }
        return render(request, "index2.html", ctx)

    ctx = { 'img_path': 'input_img.png',
            'do_success': -1, # -1 表明不需要对用户进行提示
            'view1': 'view1.png',
            'view2': 'view2.png',
            'key': 1}
    
    return render(request, 'index2.html', ctx)

def show_multi_features(request):
    """显示多层级特征"""

    global model
    if model is None:
        ctx = {
            'img_path': 'input_img.png',
            'do_success': 0,
            'key': 0,
        }
        return render(request, "index2.html", ctx)
    ctx = { 'img_path': 'input_img.png',
            'do_success': -1, # -1 表明不需要对用户进行提示
            "low_v1": 'low_v1.png',
            "mid_v1": 'mid_v1.png',
            "v1_level3": 'high_v1.png',
            "low_v2": 'low_v2.png',
            "mid_v2": 'mid_v2.png',
            "v2_level3": 'high_v2.png',
            "key": 2}
    return render(request, 'index2.html', ctx)

def show_attention_map(request):
    """显示注意力图"""

    global model
    if model is None:
        ctx = {
            'img_path': 'input_img.png',
            'do_success': 0,
            'key': 0,
        }
        return render(request, "index2.html", ctx)
    ctx = {'img_path': 'input_img.png',
            'do_success': -1, # -1 表明不需要对用户进行提示
           'hotmap': 'grad_cam.png',
           'key': 3, 
            }
    return render(request, 'index2.html', ctx)
