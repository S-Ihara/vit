import os
import argparse
import datetime

from clearml import Task
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from modules.VisionTransformer import VisionTransformer, MaskedAutoencoderViT
from configs import Default

to_np = lambda x: x.to('cpu').detach().numpy().copy()

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', type=str, default='DefaultConfigs', help='config file')
args = parser.parse_args()

### init
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# clearml
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
project_name = "vit_test"
task_name = "test"+"_"+now
tag = "vit"
comment = """
    mae vit
"""

task = Task.init(project_name=project_name, task_name=task_name)
task.add_tags(tag)
task.set_comment(comment)
logger = task.get_logger()

# ハイパーパラメータ
config = Default()
task.connect(config._asdict(), name="config")

### データセット
# データセットの読み込み
root_dir = "/mnt/local/datasets/"

train_dataset = datasets.CIFAR10(
    root=root_dir, train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    )

test_dataset = datasets.CIFAR10(root=root_dir, 
    train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    )

n_trains = len(train_dataset)
train_size = int(n_trains * 0.9)
valid_size = n_trains - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

# データローダー
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=config.pretrain_batch_size,shuffle=True,num_workers=config.num_worker)
validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.pretrain_batch_size,shuffle=False,num_workers=config.num_worker)
testloader  = torch.utils.data.DataLoader(test_dataset,batch_size=config.pretrain_batch_size,shuffle=False,num_workers=config.num_worker)

### モデル

model = MaskedAutoencoderViT(
    image_size=config.image_size,
    patch_size=config.patch_size,
    in_channel=3,
    dim=config.dim,
    hidden_dim=config.dim * 4,
    num_heads=config.num_heads,
    activation=config.activation,
    num_blocks=config.num_blocks,
    qkv_bias=config.qkv_bias,
    dropout=config.dropout,
    decoder_dim=config.dim // 2,
    decoder_num_blocks=config.decoder_num_blocks, 
    decoder_num_heads=config.decoder_num_heads,
    quiet_attention=config.quiet_attention,
)
model = model.to(device)

## 事前学習
#optimizer = optim.AdamW(model.parameters(),lr=config.pretrain_lr,betas=(0.9,0.95),weight_decay=0.05)
optimizer = optim.Adam(model.parameters(),lr=config.pretrain_lr, betas=(0.9,0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.pretrain_epochs, eta_min=config.pretrain_lr*0.1)
for epoch in tqdm(range(config.pretrain_epochs)):
    model.train()
    losses = []
    for i,data in enumerate(trainloader):
        inputs,_ = data
        inputs = inputs.to(device)

        loss,pred,mask = model(inputs,mask_ratio=config.mask_ratio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        scheduler.step()

    loss = sum(losses) / len(losses)
    logger.report_scalar(title="pretrain_loss",series="loss",value=loss,iteration=epoch)

    # valid
    model.eval()
    losses = []
    for i,data in enumerate(validloader):
        inputs,_ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            loss,pred,mask = model(inputs,mask_ratio=config.mask_ratio)

        losses.append(loss.item())
    logger.report_scalar(title="pretrain_loss",series="valid_loss",value=sum(losses)/len(losses),iteration=epoch)

# check 
model.eval()
inputs,_ = next(iter(validloader))
inputs = inputs.to(device)
loss,pred,mask = model(inputs,mask_ratio=config.mask_ratio)

pred = model.unpatchify(pred)
pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
mask = mask.detach()
mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embedding.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
inputs = torch.einsum('nchw->nhwc', inputs)

im_masked = inputs.detach().cpu() * (1 - mask)
im_paste = inputs.detach().cpu() * (1 - mask) + pred * mask

# denormalize
im_masked = im_masked * 0.5 + 0.5
im_paste = im_paste * 0.5 + 0.5
im_masked = torch.clamp(im_masked, 0., 1.)
im_paste = torch.clamp(im_paste, 0., 1.)

for i in range(8):
    logger.report_image(title="pretrain",series=f"im_masked_{i}",image=to_np(im_masked[i]),iteration=0)
    logger.report_image(title="pretrain",series=f"im_paste_{i}",image=to_np(im_paste[i]),iteration=0)

model.to_classifier_model(num_classes=len(test_dataset.classes))
model = model.to(device)

# メモリ削減のために不要な変数を削除
del pred,mask,inputs,im_masked,im_paste,_,data,losses,loss
torch.cuda.empty_cache()

trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.num_worker)
validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker)
testloader  = torch.utils.data.DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker)

# main学習
### 学習設定
# 損失関数
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=config.lr*0.1)

### 訓練ループ
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct_cnt = 0
    total_cnt = 0
    for i, data in enumerate(tqdm(trainloader, 0)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        correct_cnt += (predicted == labels).sum().item()
        
        running_loss += loss.item()
    scheduler.step()
    train_loss = running_loss / (i+1)
    train_losses.append(train_loss)
    
    # valid
    model.eval()
    valid_correct_cnt = 0
    valid_total_loss = 0
    for i,data in enumerate(validloader):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            valid_correct_cnt += (predicted == labels).sum().item()
            
            valid_total_loss += loss.item()
    valid_loss = valid_total_loss / (i+1)
    valid_losses.append(valid_loss)

    train_accs.append(correct_cnt/train_size)
    valid_accs.append(valid_correct_cnt/valid_size)
    print(f"epoch: {epoch} \n train_loss: {train_loss} train_acc: {correct_cnt/train_size*100}")
    print(f" valid_loss: {valid_loss} valid_acc: {valid_correct_cnt/valid_size*100}")

    logger.report_scalar(title="train_loss",series="loss",value=running_loss,iteration=epoch)
    logger.report_scalar(title="train_acc",series="acc",value=correct_cnt/train_size*100,iteration=epoch)
    logger.report_scalar(title="valid_loss",series="loss",value=valid_loss,iteration=epoch)
    logger.report_scalar(title="valid_acc",series="acc",value=valid_correct_cnt/valid_size*100,iteration=epoch)
    logger.report_scalar(title="lr",series="lr",value=scheduler.get_last_lr()[0],iteration=epoch)

print("Finished Training")