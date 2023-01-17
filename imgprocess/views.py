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
# Create your views here.
def index(request):
    """主页面
    """
    upload_img = request.session.get('input_img','')
    view1 = request.session.get('view1','')
    view2 = request.session.get('view2','')
    v1_level1 = request.session.get('v1_level1','')
    v1_level2 = request.session.get('v1_level2','')
    v1_level3 = request.session.get('v1_level3','')
    v2_level1 = request.session.get('v2_level1','')
    v2_level2 = request.session.get('v2_level2','')
    v2_level3 = request.session.get('v2_level3','')

    v_level_list = [
        v1_level1,
        v1_level2,
        v1_level3,
        v2_level1,
        v2_level2,
        v2_level3,
    ]
    hotmap = request.session.get('hotmap','')
    if upload_img == '':
        upload_img = 'cat1.png'

    if view1 == '' or view2 == '':
        view1 = "default.jpg"
        view2 = "default.jpg"
    
    if v1_level1 == '' or v1_level2 == '' or v1_level3 == '' or v2_level1 == '' or v2_level2 == '' or v2_level3 =='':
        v1_level1 = 'default.jpg'
        v1_level2 = 'default.jpg'
        v1_level3 = 'default.jpg'
        v2_level1 = 'default.jpg'
        v2_level2 = 'default.jpg'
        v2_level3 = 'default.jpg'

    # for v in v_level_list:
    #     if v=='':
    #         v = 'sea.jpg'
    
    
    # del request.session['img_path']
    # del request.session['view1']
    # del request.session['view2']

    content = {"img_path":upload_img,
                "view1": view1,
                "view2": view2,
                
                "hotmap": "goldfish.png"}

    return render(request, "index.html", content)

def index2(request):
    """主页面
    """
    content = {"img_path":'input_img.png',
                "key": 0}

    return render(request, "index2.html", content)



def pic_handle(request):
    """保存并处理图片
    """

    f1=request.FILES.get('pic')

    if f1 is None:
        request.session['msg'] = ' 图像不能为空'
        return redirect(index) 
    
    input_img = "input_img.png"
    fname_media='./media/input_img.png'

    with open(fname_media,'wb+') as pic:    
        for c in f1.chunks():
            pic.write(c)

    request.session["msg"] = " 上传成功！"

    return redirect(index2)


def show_result_18(request):
    """显示图像处理结果
    """
    # 对图片进行处理：模型预测，中间图像保存
    input_imgp = './media/input_img.png'
    v1, v2 = get_views(input_imgp,image_size=448)

    request.session['view1'] = 'v1.png'
    request.session['view2'] = 'v2.png'

    model = get_backbone("resnet18_ml_backbone",castrate=False)
    eval_from = './deepM/mlrl-fusion-tiny_imagenet-ep200-200-resnet18_ml_backbone.pth'
    save_dict = torch.load(eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    
    x_v1, low_v1, mid_v1, high_v1 = model.forward(v1.view((1,3,448,448)))
    x_v2, low_v2, mid_v2, high_v2 = model.forward(v2.view((1,3,448,448)))

    get_level_feature_map(low_v1, low_v2, img_prefix="low", t=2, gap=4, num_p=3)
    get_level_feature_map(mid_v1, mid_v2, img_prefix="mid", t=10, gap=3, num_p=6)
    get_level_feature_map(high_v1, high_v2, img_prefix="high", t=15, gap=1, num_p=12)

    get_cam(img_path=input_imgp)


    # request.session['v1_level1'] = 'low_v1.png'
    # request.session['v1_level2'] = 'mid_v1.png'
    # request.session['v1_level3'] = 'high_v1.png'
    # request.session['v2_level1'] = 'low_v2.png'
    # request.session['v2_level2'] = 'mid_v2.png'
    # request.session['v2_level3'] = 'high_v2.png'
    request.session['deal_success'] = '图像处理完成'

    return redirect(index2)

def show_result_50(request):
    """显示图像处理结果
    """

    return redirect(index)

def show_classification_result(request):
    """显示分类结果
    """
    ...

def show_enhanced_views(request):
    """显示增强后视图"""

    ctx = { 'img_path': 'input_img.png',
            'view1': 'view1.png',
            'view2': 'view2.png',
            'key': 1}
    
    return render(request, 'index2.html', ctx)

def show_multi_features(request):
    """显示多层级特征"""
    ctx = { 'img_path': 'input_img.png',
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
    ...
