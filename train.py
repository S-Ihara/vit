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

# seed 
def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    #torch.use_deterministic_algorithms(True)
SEED = 42
G = torch.Generator()
G.manual_seed(SEED)
set_seed(SEED)

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
    scratch vit
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

# transforms.RandomHorizontalFlip()


train_dataset = datasets.CIFAR10(
    root=root_dir, train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomInvert(p=0.5),
                                  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                  transforms.RandomErasing(p=0.25),
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
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=G)

# データローダー
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.num_worker,generator=G,worker_init_fn=set_seed(42))
validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker,generator=G,worker_init_fn=set_seed(42))
testloader  = torch.utils.data.DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker,generator=G,worker_init_fn=set_seed(42))

### モデル
model = VisionTransformer(
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
    quiet_attention=config.quiet_attention,
    num_classes=10,
)
model = model.to(device)

### 学習設定
# 損失関数
criterion = nn.CrossEntropyLoss().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

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