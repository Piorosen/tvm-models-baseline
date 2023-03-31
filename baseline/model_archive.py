import torch
import torchvision

# set the model to inference mode 



def resnet18():
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.to("cpu")
    model.eval()
    return model

def resnet50():
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model.to("cpu")
    model.eval()
    return model

def mobilenetv2():
    mbn2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    mbn2.eval() 
    return mbn2
    
def vgg16():
    vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    vgg.eval() 
    return vgg
    
def googlenet():
    goog = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    goog.eval() 
    return goog
    
def alexnet():
    alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    alex.eval() 
    return alex

MODEL_ARCHIVE = {
    "alexnet": {
        "model": alexnet,
        "input": (torch.randn(1, 3, 227, 227),),
    },
    "vgg16": {
        "model": vgg16,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "googlenet": {
        "model": googlenet,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "mobilenetv2": {
        "model": mobilenetv2,
        "input": (torch.randn(1, 3, 224, 224),),
    },
}
