import math
import io
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pybboxes as pbx
import json
from pybboxes import BoundingBox
from sklearn.metrics import average_precision_score
import numpy
import matplotlib.pyplot as plt
import os
from scipy import integrate
from numpy import trapz
import random
import cv2
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()
       
import torch.nn.functional as F
def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return padding_left, padding_bottom, F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    
    
def pillow_encode(img, fmt='jpeg', quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp).convert("RGB")
    return rec, bpp

def find_closest_bpp(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp
    


from compressai.zoo import models as pretrained_models



def img_compress(adr, net, quality):
    img = Image.open(adr).convert('RGB')
    h, w = img.size
    #x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    h_0, w_0, x = pad(transforms.ToTensor()(img).unsqueeze(0).to(device))
    with torch.no_grad():
        out_net = net.forward(x)
    out_net['x_hat'].clamp_(0, 1) 
    rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu()).crop((h_0, w_0, h_0 + h, w_0 + w))
    l = len(adr)
    #change path if need to store compressed pictures
    #str1 = f'/home/dariats/Downloads/darknet/coco/images2014_compressed/quality{quality}.png'
    #rec_net = rec_net.save(str1)
    bpp = round(compute_bpp(out_net), 3)
    return rec_net, bpp

def net_res_dl(input_image, model):
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    return output
    
def net_res_mb(input_image, model):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    pred = []
    for i in range(top5_prob.size(0)):
        pred.append([categories[top5_catid[i]], top5_prob[i].item()])
        
    return pred
    
def iou(object_tr, obj, image_size):
    xmin = obj['xmin']
    ymin = obj['ymin']
    xmax = obj['xmax']
    ymax = obj['ymax']
    voc = (xmin, ymin, xmax, ymax)
    voc_bbox = pbx.convert_bbox(voc, from_type="voc", to_type="coco", image_size=image_size, strict = False)
    coco_bbox = BoundingBox.from_coco(*voc_bbox, image_size=image_size, strict = False)
    xmin = object_tr['xmin']
    ymin = object_tr['ymin']
    xmax = object_tr['xmax']
    ymax = object_tr['ymax']
    voc_tr = (xmin, ymin, xmax, ymax)
    voc_bbox_tr = pbx.convert_bbox(voc_tr, from_type="voc", to_type="coco", image_size=image_size, strict = False)
    coco_bbox2 = BoundingBox.from_coco(*voc_bbox_tr, image_size=image_size, strict = False)
    iou = coco_bbox.iou(coco_bbox2)
    return iou

def av_iou_y(true_objects, objects, image_size):
    conf_tresh = 0.4
    iou_tresh = 0.5
    """while i <(len(true_objects):
        if true_objects[i]["confidence"] < conf_tresh:
            true_objects.pop(i)
        else:
            i+=1"""      
    avg_iou = 0
    iou_count = 0
    j = 0
    while j < len(objects):
        object1 = objects[j]
            
        """"if object1['confidence'] < conf_tresh:
            continue"""
        max_iou = 0
        i = 0
        pos_tr = 0
        max_tr_obj = {}
        for object_tr in true_objects:
            if (object1['name'] != object_tr['name']):
                i+= 1
                continue                                 
            iou1 = iou(object_tr, object1, image_size)
            if iou1 > max_iou:
                pos_tr = i
                max_iou = iou1
                max_tr_obj = object_tr
            i+=1
        if max_tr_obj == {}:
            j+=1
            continue
        i = 0
        pos = 0
        max_iou = 0
        for obj in objects:
            if (obj['name'] != max_tr_obj['name']):
                i+= 1
                continue                                 
            iou1 = iou(max_tr_obj, obj, image_size)
            if iou1 > max_iou:
                pos = i
                max_iou = iou1
            i+= 1
        
        #print(objects[pos], true_objects[pos_tr])
        
        
        true_objects.pop(pos_tr)
        objects.pop(pos)
        if pos > j:
            j+=1
        if (max_iou > iou_tresh):
            avg_iou+=max_iou
            iou_count+=1

    if (iou_count+len(true_objects) > 0):   
        avg_iou = avg_iou/(iou_count+len(true_objects))
    return avg_iou
    
def f5_accur(true_res, res):
    a = [0.5, 0.25, 0.125, 0.0625, 0.0625]
    accur = 0
    for i in range(5):
        for j in range(5):
            if res[i][0] == true_res[j][0]:
                d = res[i][1]/true_res[j][1]
                if (d > 1):
                    d = 1/d
                k = i+abs(i-j)
                if k >4: 
                    k = 4
                accur += d*a[k]
                break
    return accur


def av_iou_dl(tr_tensor, tensor):
    summ = 0
    amount = 0
    for i in range(1, 21):
        iou = 0
        temp_tr_tensor = copy.deepcopy(tr_tensor)
        temp_tensor = tensor
        temp_tr_tensor[temp_tr_tensor != i] = 0
        temp_tr_tensor[temp_tr_tensor == i] = 1
        
        if torch.sum(temp_tr_tensor) != 0:
            amount+=1
            temp_tensor[temp_tensor != i] = 0
            temp_tensor[temp_tensor == i] = 1
            ten_and = (torch.logical_and(temp_tr_tensor, temp_tensor)).long()
            ten_or = (torch.logical_or(temp_tr_tensor, temp_tensor)).long()
            iou = torch.sum(ten_and)/torch.sum(ten_or)
            summ +=iou
    if amount != 0:
        summ = summ/amount
    if summ ==0:
        return 0.0
    return summ.item()
        
    
def compute( adr, model_dl, model_y, model_mb):
    
    names = ['bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn', 'webp','jpeg']
    qual = {'bmshj2018-factorized':9,
            'bmshj2018-hyperprior':9,
            'mbt2018-mean':9,
            'mbt2018':9,
            'cheng2020-anchor':7,
           'cheng2020-attn':7,
           'webp':9,
           'jpeg':9}
    data = {}
    data['deeplabv3'] = {}
    data['yolov5'] ={}
    data['mobilenet_v2'] = {}   
    data_dl = {}
    data_y = {}
    data_mb = {}
    image = Image.open(adr).convert("RGB")
    true_results_dl = net_res_dl(image, model_dl)

    tr_results_mb = net_res_mb(image, model_mb)
    data_mb["true objects"] = tr_results_mb
    tr_results_y = model_y(image)
    tr_results_y = tr_results_y.pandas().xyxy[0].to_dict('records')
    data_y["true objects"] = tr_results_y
    for name in names:
        bpps = []
        ious_dl = []
        ious_y = []
        accurs_mb = []
        data_dl[name] ={}
        data_mb[name] = {}
        data_y[name] = {}
        for quality in range(1,qual[name]):
            if name == 'webp' or name == 'jpeg':
                target_bpp = target_bpps[quality - 1]
                img = Image.open(adr).convert('RGB')
                image, bpp = find_closest_bpp(target_bpp, img,fmt=name)
            else:
                net = pretrained_models[name](quality=quality, pretrained=True).eval().to(device)
                image, bpp =img_compress(adr, net, quality)
            
            bpps.append(bpp)
            
            results_dl = net_res_dl(image,model_dl)
            ious_dl.append(av_iou_dl(true_results_dl.argmax(0), results_dl.argmax(0)))
            
            results_y = model_y(image)
            results_y = results_y.pandas().xyxy[0].to_dict('records')
            data_y[name][quality]  = results_y
            iou_y = av_iou_y(tr_results_y[:], results_y[:], image.size)
            ious_y.append(iou_y)
            
            results_mb = net_res_mb(image, model_mb)
            data_mb[name][quality]  = results_mb
            accurs_mb.append(f5_accur(tr_results_mb, results_mb))
            
        data_y[name]["curves"] = {}
        data_y[name]["curves"] = [bpps, ious_y]
        
        data_dl[name]["curves"] = {}
        data_dl[name]["curves"] = [bpps, ious_dl]
        
        data_mb[name]["curves"] = {}
        data_mb[name]["curves"] = [bpps, accurs_mb]
        
        x = bpps[len(bpps) - 1]- bpps[0]
        data_y[name]["integrate"] = integrate.trapz(ious_y, bpps)/x
        data_dl[name]["integrate"] = integrate.trapz(ious_dl, bpps)/x
        data_mb[name]["integrate"] = integrate.trapz(accurs_mb, bpps)/x
        
        if name == 'webp':
            bpps = bpps[:6]
            ious_y  = ious_y[:6]
            accurs_mb = accurs_mb[:6]
            ious_dl = ious_dl[:6]    
            x = bpps[len(bpps) - 1]- bpps[0]
            data_y[name]["integrate2"] = integrate.trapz(ious_y, bpps)/x
            data_dl[name]["integrate2"] = integrate.trapz(ious_dl, bpps)/x
            data_mb[name]["integrate2"] = integrate.trapz(accurs_mb, bpps)/x
        
        
        if name == 'mbt2018':
            target_bpps = bpps
    
    
    data['deeplabv3'] =data_dl
    data['yolov5'] =data_y
    data['mobilenet_v2'] = data_mb   
    
    return data
    
#if name == "__main__":
model_dl = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model_dl.eval()
model_y =  torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_mb = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model_mb.eval()
#path to file with image adresses    
f = open('/home/dariats/Downloads/darknet/coco/in_process.txt', 'r')
for line in f:
    file = f'/home/dariats/Downloads/darknet_new/img_results/val_results/{line[-30:-5]}.json'
    adr = line[:-1]
    res_data = compute(adr, model_dl, model_y, model_mb)
    with open(file, 'w') as outfile:
        json.dump(res_data, outfile)
    print(line[-30:-5])
f.close()
