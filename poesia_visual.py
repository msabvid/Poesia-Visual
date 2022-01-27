import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse
import os
import tqdm


# congelo els parametres de la xarxa. 
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


# loader i unloader


def image_loader(image_name, device):
    image = Image.open(image_name)
    width, height = image.size
    if device=='cpu':
        s = 128
    else:
        s = 256
    loader = transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0) # creo batch dim
    return image.to(device).float(), (width, height)

def unloader(H, W):
    return transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((H, W)),
                              ])


def imshow(tensor, height, width, title=None):
    image = tensor.cpu().clone()  # 
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(H=height, W=width)(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

# content loss
class ContentLoss(nn.Module):

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        # oju que es molt sublim aixo: intercalo ContentLoss a cada capa de la xarxa, per tant hem de retornar l'input, i no la loss
        self.loss = F.mse_loss(input, self.target)
        return input

# style loss
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we normalize the values of the gram matrix
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, style_img, content_img, device='cpu'):
                           
    cnn = copy.deepcopy(cnn)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # normalization module
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # creo el model incrementalment
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Netegem les ultimes layers de la xarxa que no ens interessen
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def main(path_style: str, path_content: str, path_result: str,
         num_steps: int,
         style_weight: float,
         content_weight: float,
         from_white_noise: bool,
         device: str
        ):
    """
    Funcio main que fa tot el tema:
        1) Llegeix i transforma les imatges
        2) Carrega el model ja entrenat (vgg) i calcula els diferents valors per la loss style i la loss function
        3) Optimitza! (utilitzo L-BFGS optimitzador)
        4) Flipa amb el resultat
    """
    # get network
    device = device
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    freeze(cnn)
    # get content, style and input images
    style_img, _ = image_loader(path_style, device=device)
    content_img, (W,H) = image_loader(path_content, device=device)
    if from_white_noise:
        input_img = torch.randn_like(content_img)
    else:
        input_img = content_img.clone()
    
    # optimitzem l'input i no els parametres de la xarxa
    print('Building the model')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, device=device)

    print('Optimizing...')
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    pbar = tqdm.tqdm(total = num_steps)
    for idx in range(num_steps):
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            if (idx+1) % 10 == 0:
                pbar.write('Style Loss : {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)
        pbar.update(1)
    
    #input_img.data = (input_img.data - input_img.data.min()) / (input_img.data.max() - input_img.data.min())
    plt.figure()
    imshow(input_img, height=H, width=W)
    plt.savefig(path_result)







if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--style_image', type=str)
    parser.add_argument('--content_image', type=str)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--style_weight', default=1e6, type=float)
    parser.add_argument('--content_weight', default=1., type=float)
    parser.add_argument('--from_white_noise', action='store_true', default=False, 
            help='crea la nova imatge de soroll gaussia')
    
    args = parser.parse_args()

    path_style = os.path.join('style', args.style_image)
    if not os.path.exists(path_style):
        raise ValueError('La imatge {} no existeix'.format(path_style))
    path_content = os.path.join('content', args.content_image)
    if not os.path.exists(path_content):
        raise ValueError('La imatge {} no existeix'.format(path_content))
    
    filename_result = '_'.join((
            os.path.splitext(args.content_image)[0],
            os.path.splitext(args.style_image)[0])
            ) + '.jpg'
    if not os.path.exists('resultat-poesia-visual'):
        os.makedirs('resultat-poesia-visual')
    path_result = os.path.join('resultat-poesia-visual', filename_result)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    config = vars(args)
    config.pop('style_image')
    config.pop('content_image')
    config.update(device=device)
    main(path_style, path_content, path_result, **config)


# <3
