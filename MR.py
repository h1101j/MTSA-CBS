import os
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from model.model_main_fr import FRQA
from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
from option.config import Config
from trainer_fr import train_epoch_fr, test_epoch_fr
from util1 import RandCrop, RandHorizontalFlip, RandRotation, Normalize, ToTensor, RandShuffle
from data.data_pre import QADataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from data_loader_nr import DataLoader_n
import model_final_nr
from utils import calc_coefficient, lr_scheduler
from utils import save_checkpoint, load_checkpoint
import model_pretrain_nr
import csv
from scipy.stats import spearmanr, pearsonr
import itertools
import math
from sklearn.metrics import r2_score

# config file
config = Config({
    # device
    "GPU_ID": "1",
    "num_workers": 8,

    "n_enc_seq": 25*25,               # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 25*25,               # feature map dimension (H x W) from backbone
    "n_layer": 2,                     # number of encoder/decoder layers
    "d_hidn": 256,                    # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,                     # feed forward hidden layer dimension
    "d_MLP_head": 512,                # hidden layer of final MLP
    "n_head": 4,                      # number of head (in multi-head attention)
    "d_head": 256,                    # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,                   # dropout ratio of transformer
    "emb_dropout": 0.1,               # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,                    # dimension of final prediction
    "crop_size": 224,                 # input image crop size

    # data
    "db_name": "cbs",
    "db_path": "...",                          # root of FRF dataset
    "snap_path": "./weights/cbs",              # path for saving FR moudle weights
    "txt_file_name": "...",                    # image list file (.txt)
    "train_size": 0.8,
    "scenes": "all",

    #NR optimization
    "CKPT_F": "...",
    "CKPT_P": "...",
    "MODEL_PATH_F": "save_models_pretrain",
    "MODEL_PATH_P": "save_models_score",
    "SEED": 0,
    "PATCH_SIZE": 224,
    "TRAIN_PATCH_NUM": 1,
    "TEST_PATCH_NUM": 1,
    "BATCH_SIZE": 10,
    "LEARNING_RATE_F": 2,

    "test_ensemble": False,
    "n_ensemble": 5,

    # FR optimization
    "batch_size": 15,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,
    "n_epoch": 100,
    "val_freq": 1,
    "save_freq": 5,
    "checkpoint": None,                 # load pretrained weights
    "T_max": 50,                        # cosine learning rate period (iteration)
    "eta_min": 0                        # mininum learning rate
})

def train_nr(optimizer1, best_srocc, coef, corre_plcc, best_model, best_optimzer):
    losses = []
    print(f'+====================+ Training Epoch: {epoch} +====================+')
    loop = tqdm(dataloader_train)
    optimizer1 = lr_scheduler(optimizer1, epoch)
    snr_train = []

    for batch_idx, (dist, rating) in enumerate(loop):
        batch_size = dist.shape[0]
        dist = dist.to(config.device).float()
        rating = rating.reshape(batch_size, -1).to(config.device).float()

        emap, _ = model_err(dist)
        inp = (dist, emap)

        optimizer1.zero_grad()
        out, attn_list = model(inp)
        snr_train.append(out)
        loss = loss_fn(out, rating)

        loss.backward()
        optimizer1.step()
        losses.append(loss)
        loop.set_postfix(loss=loss.item())

    print(f'Loss: {sum(losses)/len(losses):.5f}')
    print(f'+====================+ Testing Epoch: {epoch} +====================+')
    sp, pl, snr = calc_coefficient(dataloader_test, model, config.device, model_err)
    print(f'SROCC: {sp:.3f}, PLCC: {pl:.3f}')

    # save models
    if sp > best_srocc:
        best_srocc = sp
        corre_plcc = pl
        best_model = model
        best_optimzer = optimizer1
    print(f'BEST SROCC: {best_srocc:.3f} & PLCC: {corre_plcc:.3f}')
    coef['srocc'], coef['plcc'] = best_srocc, corre_plcc
    snr_train_cpu = []
    for tensor in snr_train:
        snr_train_cpu.append(tensor.cpu())
    snr_train_cpu = [tensor.detach().numpy() for tensor in snr_train_cpu]
    snr_train_cpu = np.array(snr_train_cpu).flatten()
    return coef, best_model, best_optimzer, snr, snr_train_cpu

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out

    # device setting
config.device = torch.device("cuda:%s" %config.GPU_ID if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % config.GPU_ID)
else:
    print('Using CPU')

# data separation (8:2)
train_scene_list, test_scene_list = RandShuffle(config.scenes, config.train_size)
print('number of train scenes: %d' % len(train_scene_list))
print(train_scene_list)
print('number of test scenes: %d' % len(test_scene_list))
print(test_scene_list)

# FR data load
train_dataset = QADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform=transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), RandHorizontalFlip(), RandRotation(), ToTensor()]),
    train_mode=True,
    scene_list=train_scene_list,
    train_size=config.train_size
)
test_dataset = QADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform= transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), ToTensor()]) if config.test_ensemble else transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), ToTensor()]),
    train_mode=False,
    scene_list=test_scene_list,
    train_size=config.train_size
)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=False)


# create FR model
model_transformer = FRQA(config).to(config.device)
model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background').to(config.device)

# save intermediate layers
save_output = SaveOutput()
hook_handles = []
for layer in model_backbone.modules():
    if isinstance(layer, Mixed_5b):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
    elif isinstance(layer, Block35):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

# loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# NRF data
folder_path = '...'

if not os.path.exists(config.MODEL_PATH_F):
    os.mkdir(config.MODEL_PATH_F)

# fix the seed if needed for reproducibility
if config.SEED == 0:
    pass
else:
    print('SEED = {}'.format(config.SEED))
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

train_index = train_scene_list
test_index = test_scene_list

# build NR train and test loader
dataloader_train = DataLoader_n('3d',
                            folder_path,
                            train_index,
                            config.PATCH_SIZE,
                            config.TRAIN_PATCH_NUM,
                            config.BATCH_SIZE,
                            istrain=True).get_data()

dataloader_test = DataLoader_n('3d',
                            folder_path,
                            test_index,
                            config.PATCH_SIZE,
                            config.TEST_PATCH_NUM,
                            config.BATCH_SIZE,
                            istrain=False).get_data()


model = model_final_nr.vit_IQAModel(pretrained=True).to(config.device)

# load pretrain model
load_path = config.CKPT_P.format('30')
load_path = os.path.join(config.MODEL_PATH_P, load_path)
ckpt = torch.load(load_path, map_location=config.device)
model_err = model_pretrain_nr.vit_IQAModel().to(config.device)
model_err.load_state_dict(ckpt['state_dict'])
model_err.requires_grad_(False)

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
optimizer1 = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_F)

model.train()

# train & test
losses, scores = [], []
srocc_max = 0
best_srocc = 0
coef = {}
corre_plcc = 0
best_model = None
best_optimzer = None

input_dim = 2
hidden_dim = 16
output_dim = 1

modelmlp = MLP(input_dim, hidden_dim, output_dim).to(config.device)
criterionmlp = nn.MSELoss()
optimizermlp = optim.Adam(modelmlp.parameters(), lr=0.001)

sstrain = []
sstest = []

# load scene score(.csv)
with open('....', mode='r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) > 1:
            if row[1] == '':
                continue
            if int(row[0]) in train_scene_list:
                sstrain.append(float(row[1]))
            if int(row[0]) in test_scene_list:
                sstest.append(float(row[1]))

for epoch in range(0, config.n_epoch):
    loss, rho_s, rho_p, sfr_train = train_epoch_fr(config, epoch, model_transformer, model_backbone, save_output, criterion, optimizer, scheduler, train_loader)
    if (epoch+1) % config.val_freq == 0:
        loss, rho_s, rho_p, sfr_test = test_epoch_fr(config, epoch, model_transformer, model_backbone, save_output, criterion, test_loader)
    coef, model, optimizer1, snr_test, snr_train = train_nr(optimizer1, best_srocc, coef, corre_plcc, best_model, best_optimzer)
    if coef['srocc'] > srocc_max:
        srocc_max = coef['srocc']
        save_path = config.CKPT_F.format(epoch)
        save_path = os.path.join(config.MODEL_PATH_F, save_path)
        save_checkpoint(model, optimizer1, filename=save_path)

    synthesized_scores_150_train = [np.mean(sfr_train[i * 150:(i + 1) * 150]) for i in range(len(sfr_train) // 150)]
    synthesized_scores_50_train = [np.mean(snr_train[i * 50:(i + 1) * 50]) for i in range(len(snr_train) // 50)]
    scores_tensor_150_train = torch.tensor(synthesized_scores_150_train, dtype=torch.float32).view(-1, 1)
    scores_tensor_50_train = torch.tensor(synthesized_scores_50_train, dtype=torch.float32).view(-1, 1)

    synthesized_scores_150 = [np.mean(sfr_test[i * 150:(i + 1) * 150]) for i in range (len(sfr_test) // 150)]
    synthesized_scores_50 = [np.mean(snr_test[i * 50:(i + 1) * 50]) for i in range (len(snr_test) // 50)]
    scores_tensor_150 = torch.tensor(synthesized_scores_150, dtype=torch.float32).view(-1, 1)
    scores_tensor_50 = torch.tensor(synthesized_scores_50, dtype=torch.float32).view(-1, 1)

    target_scores = torch.tensor(sstrain, dtype=torch.float32).view(-1, 1).to(config.device)
    test_scores = torch.tensor(sstest, dtype=torch.float32).view(-1, 1).to(config.device)

    combined_scores_train = torch.cat((scores_tensor_150_train, scores_tensor_50_train), dim=1).to(config.device)
    combined_scores = torch.cat((scores_tensor_150, scores_tensor_50), dim=1).to(config.device)

    modelmlp.train()
    optimizermlp.zero_grad()
    outputs = modelmlp(combined_scores_train)
    lossmlp = criterionmlp(outputs, target_scores)
    lossmlp.backward()
    optimizermlp.step()

    test_output = modelmlp(combined_scores)
    test_output = test_output.cpu()
    test_output_cpu = test_output.detach().numpy()
    test_output_cpu = test_output_cpu.tolist()
    test_output_cpu = list(itertools.chain(*test_output_cpu))

    print('scenes')
    ssrcc, _ = spearmanr(test_output_cpu, sstest)
    splcc, _ = pearsonr(test_output_cpu, sstest)
    print('scene srcc:', ssrcc)
    print('scene plcc:', splcc)


