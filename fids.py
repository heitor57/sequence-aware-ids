# %% [markdown]
# # Packages Installs, imports, and presets

# %%
#from scapy.all import *
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict
import os
from transformers import MobileViTV2Model, MobileViTV2Config
import torch
import torchmetrics
from torch import optim
import lightning as L
import torch
import torch.nn as nn

# %%
INPUT_SIZE = (256,256,3)

# %%
df = pd.read_parquet('data/iec104_15.parquet')

# %% [markdown]
# ## Removing missing features

# %%
df=df.drop([c for c,v in ((df==-1).sum() == len(df)).items() if v==True],axis=1)

df['tv_sec']=df['tv_sec'].astype(int)
df['tv_usec']=df['tv_usec'].astype(int)

df[['label']].value_counts()

# %%
df[['label']]

# %%
from feature_engine.encoding import OrdinalEncoder

# %%
od = OrdinalEncoder(encoding_method='arbitrary')
od.fit(df[['label']])
df[['label']]= od.transform(df[['label']])

# %%
TRAIN_RATE= 0.8
import random
indexes = list(set(df.index))
random.shuffle(indexes)
TRAIN_SIZE = int(len(indexes)*TRAIN_RATE)
TEST_SIZE = len(indexes)-TRAIN_SIZE
TRAIN_SAMPLES = indexes[:TRAIN_SIZE]
TEST_SAMPLES = indexes[:TEST_SIZE]
df_train_initial = df.loc[TRAIN_SAMPLES]
df_test = df.loc[TEST_SAMPLES]
# df_test = df.iloc[TRAIN_SAMPLES:]
# del df

# %%
indexes = list(set(df_train_initial.index))
random.shuffle(indexes)
TRAIN_SIZE = int(len(indexes)*TRAIN_RATE)
TEST_SIZE = len(indexes)-TRAIN_SIZE
TRAIN_SAMPLES = indexes[:TRAIN_SIZE]
TEST_SAMPLES = indexes[:TEST_SIZE]
df_train = df_train_initial.loc[TRAIN_SAMPLES]
df_validation = df_train_initial.loc[TEST_SAMPLES]

# %%
df_train.shape, df_validation.shape, df_test.shape

# %%
len(set(df_train.index))

# %%
del df_train_initial
del df

# %%
# TRAIN_SIZE = 0.7
# tmp = random.shuffle(list(set(df_train.index)))
# TRAIN_SAMPLES = int(len(df_train)*TRAIN_SIZE)
# df_train,df_validation = df_train.iloc[:TRAIN_SAMPLES],df_train.iloc[TRAIN_SAMPLES:]

# %%
assert df_train.groupby(df_train.index)['label'].nunique().max()==1

# %% [markdown]
# #### Static vs Dynamic Packet (w.r.t. Flow) Features

# %%
# pat = '([\w\d_]+)_?\d*'
# pat=r"([\w_]+)_\d*"
pat=r"([\w_]+)_\d+"
tmp=df_train.iloc[:10000, df_train.columns != 'label'].T.reset_index().replace(to_replace=pat, value=r"\1", regex=True).groupby('index').agg(lambda x: ''.join(map(str,x))).T
tmp = tmp.groupby(tmp.index).nunique().max().to_frame().sort_values(by=0,ascending=False)

# %%
static_features= set(tmp.loc[tmp[0]==1].index)
dynamic_features = set(tmp.index) - static_features

# %%
dynamic_features_raw = {'tcp_opt','tv_sec','tv_usec','tcp_cksum','tcp_ackn','tcp_seq','ipv4_tl'}
dynamic_features_reduce =  dynamic_features - dynamic_features_raw

# %%
dynamic_features_raw,dynamic_features_reduce,static_features = list(dynamic_features_raw),list(dynamic_features_reduce),list(static_features)
dynamic_features=  list(dynamic_features)

# %%
# [i for i in df_train.columns if len([for i in dynamic_features]) > 0]
def get_bit_columns(features):
  dynamic_features_bit_columns= []
  for column_name in df_train.columns:
    for feature_name in features:
      if  feature_name in column_name:
        dynamic_features_bit_columns.append(column_name)
        break
  return dynamic_features_bit_columns

# %%
dynamic_features_bit_columns= get_bit_columns(dynamic_features)
static_features_bit_columns= get_bit_columns(static_features)

# %%
dynamic_features

# %%
num_dynamic_features = len(dynamic_features_bit_columns)
num_static_features= len(static_features_bit_columns)
num_dynamic_features, num_static_features

# %%
len(set(df_train.index))

# %% [markdown]
# ## Training Dataset
# 

# %%
import multiprocessing
multiprocessing.cpu_count()

# %%
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.utils.data import Dataset

BATCH_SIZE=32

class IEC104Dataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.flow_int_id = list(set(self.df.index))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.flow_int_id)

    def __getitem__(self, idx):
        df_idx = self.flow_int_id[idx]
        flow = self.df.loc[df_idx, self.df.columns != 'label']
        label = self.df.loc[df_idx, 'label']

        if isinstance(flow, pd.Series):
            flow = flow.to_frame().T

        if not isinstance(label, np.int64):
            label = label.iloc[0]
        label = torch.tensor(label)

        if self.transform:
            flow = self.transform(flow)
        if self.target_transform:
            label = self.target_transform(label)

        return flow, label

def feature_transform(flow):
    # start_time = time.time()
    # Convert flow to a DataFrame if it's not already
    flow = pd.DataFrame(flow)

    # Extract dynamic and static features
    dynamic_features = flow.loc[:, dynamic_features_bit_columns].values
    static_features = flow.loc[:, static_features_bit_columns].iloc[0].values

    # Convert to tensors
    dynamic_tensor = torch.tensor(dynamic_features, dtype=torch.float32)
    static_tensor = torch.tensor(static_features, dtype=torch.float32)

    # Determine the shape of the dynamic tensor
    dynamic_shape = dynamic_tensor.shape

    # Preallocate tensor for X with -1
    # print(INPUT_SIZE)
    total_dynamic_size = INPUT_SIZE[1] * INPUT_SIZE[2]
    X = -torch.ones(INPUT_SIZE[0], total_dynamic_size, dtype=torch.float32)

    # Fill in the dynamic features (ensure no size mismatch)
    min_shape_0 = min(dynamic_shape[0], INPUT_SIZE[0])

    X[:min_shape_0, :dynamic_shape[1]] = dynamic_tensor[-min_shape_0:, :dynamic_shape[1]]
    X = X.view(3,256,256)
    X = (X+1)/2
    return [X, static_tensor]

NUM_WORKERS =2
PREFETCH_FACTOR=2
train_dataset = IEC104Dataset(df_train,transform = feature_transform)
validation_dataset = IEC104Dataset(df_validation,transform = feature_transform)
test_dataset = IEC104Dataset(df_test,transform = feature_transform)

train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
validation_loader = data_utils.DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)

# %% [markdown]
# ## Model Design

# %%
model_vitv2_output_shape = [ 512, 8, 8]
NUM_CLASSES = df_train.label.nunique()

# %%
def get_device():
    device = "cpu"
    if torch.cuda.is_available():
      device = "cuda"
    return device
DEVICE = get_device()

# %%
import seaborn as sns

# %%
# from pytorch_lightning.loggers import TensorBoardLogger

# logger = TensorBoardLogger("tightning_logs", name="my_model")


# %%

class FullyConnectedNet(nn.Module):
    def __init__(self,input_size,output_size,l1=180,l2=128,l3=20,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(input_size, l1),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(l1, l2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(l2, l3),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(l3, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class FVIT(nn.Module):
    def __init__(self,encoder,decoder):
        super(FVIT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hidden_states=self.encoder(x[0]).last_hidden_state
        # print("Hidden states shape:", hidden_states.shape)
        # print("Additional input shape:", x[1].shape)
        concatenated_input = torch.cat((torch.flatten(hidden_states, start_dim=1), torch.flatten(x[1], start_dim=1)),dim=1)
        # print("Concatenated input shape:", concatenated_input.shape)
        return self.decoder(concatenated_input)
        # print(hidden_states.shape,x[1].shape)
        # return self.decoder(torch.concat((torch.flatten(hidden_states,start_dim=1), torch.flatten(x[1],start_dim=1))))


class LFVIT(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        for key,value in metrics.copy().items():
            metrics["train_"+key]=metrics.pop(key)
        self.log_dict(metrics,prog_bar=True,on_step=False, on_epoch=True)
        return metrics['train_loss']
    def forward(self, inputs):
        return self.model(inputs)
    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        for key,value in metrics.copy().items():
            metrics["val_"+key]=metrics.pop(key)
        self.log_dict(metrics,prog_bar=True,on_step=False, on_epoch=True)
        return metrics
    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        for key,value in metrics.copy().items():
            metrics["test_"+key]=metrics.pop(key)
        self.log_dict(metrics)
        return metrics
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # dynamic_features, static_features  = x
        target_prediction = self.model(x)
        # print(target_prediction.shape,y.shape)
        # print(target_prediction)
        # print(x)
        metrics=dict()
        # print(target_prediction, y)
        metrics['loss'] = nn.functional.cross_entropy(target_prediction, y)
        # Initialize metrics
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        # precision = torchmetrics.Precision(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        # recall = torchmetrics.Recall(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        # f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        # auc_roc = torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        # auc_pr = torchmetrics.AveragePrecision(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        confusionmatrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

        # Compute metrics
        metrics['accuracy'] = accuracy(target_prediction, y)
        # metrics['precision'] = precision(target_prediction, y)
        # metrics['recall'] = recall(target_prediction, y)
        # metrics['f1'] = f1(target_prediction, y)
        # metrics['auc_roc'] = auc_roc(target_prediction, y)
        # metrics['auc_pr'] = auc_pr(target_prediction, y)
        # confusionmatrix_pr = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        confusionmatrix_result = confusionmatrix(target_prediction, y).cpu().numpy()
        # print(confusionmatrix_result)
        df_cm = pd.DataFrame(confusionmatrix_result, index = range(NUM_CLASSES), columns=range(NUM_CLASSES))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        tensorboard = self.logger.experiment
        tensorboard.add_figure("confusionmatrix", fig_, self.current_epoch)
        # tensorboard.add_image("Confusion matrix", confusionmatrix_result)
        # self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

        return metrics
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
        return {'optimizer':optimizer,"lr_scheduler":{"scheduler": scheduler, "interval": "epoch","monitor": "val_loss"}}


# %%
model_vitv2 = MobileViTV2Model.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
# model_vitv2 = MobileViTV2Model(config=MobileViTV2Config())
decoder=FullyConnectedNet(np.prod(model_vitv2_output_shape)+num_static_features,NUM_CLASSES,l1=100,l2=40,l3=20,dropout=0.2)
fvit = FVIT(model_vitv2,decoder)
model_fvit = LFVIT(fvit)

# %%
# %%capture stored_output
MAX_EPOCHS=50
# timer =  L.pytorch.calalbacks.Timer()
def run_fvit_trainer_fit(train_loader,validation_loader,model):
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='train_loss',
        filename='iec104-{epoch:02d}-{loss}',
        save_top_k=1,
        mode='min',
    )

    # early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")
    # cpb=CustomProgressBar()
    trainer = L.Trainer(max_epochs=MAX_EPOCHS,callbacks=[checkpoint_callback],accelerator=DEVICE,
                        devices=1,log_every_n_steps=10,limit_train_batches=1.0,limit_val_batches=1.0,max_time={"minutes": 60*24})
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=validation_loader)
    return checkpoint_callback,trainer
checkpoint_callback,trainer=run_fvit_trainer_fit(
    train_loader,
    validation_loader,
     model=model_fvit)

# %%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# %%
def plot_training_metrics(trainer,metrics,ax):
    event_acc = EventAccumulator(trainer.logger.log_dir)
    event_acc.Reload()
    for i in range(len(metrics)):
        y=[i.value for i in event_acc.Scalars(metrics[i]['name'])]
        x=np.arange(len(y))+1
        ax[i].plot(x,y,marker='.')
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metrics[i]['label'])
metrics=[{'name':'train_loss','label':'Training MSE Loss'},{'name':'val_loss','label':'Validation MSE Loss'}]
fig,ax = plt.subplots(ncols=len(metrics),figsize=(15,4))
plot_training_metrics(trainer,metrics,ax)

# %%
metrics=[{'name':'train_accuracy','label':'Training Accuracy'},{'name':'val_accuracy','label':'Validation Accuracy'}]
fig,ax = plt.subplots(ncols=len(metrics),figsize=(15,4))
plot_training_metrics(trainer,metrics,ax)

# %%
trainer.test(ckpt_path="best",dataloaders=test_loader)

# %%
# class ImageFCN(nn.Module):
#     def __init__(self,fcn):
#         super(FVIT, self).__init__()
#         self.fcn = fcn

#     def forward(self, x):
#         return self.fcn(torch.flatten(x))
# ImageFCN(FullyConnectedNet())

# %%
# MAX_EPOCHS=50
# # timer =  L.pytorch.calalbacks.Timer()
# def run_fvit_trainer_fit(train_loader,validation_loader,model):
#     checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
#         monitor='train_loss',
#         filename='iec104-{epoch:02d}-{loss:.2f}',
#         save_top_k=3,
#         mode='min',
#     )

#     early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")
#     # cpb=CustomProgressBar()
#     trainer = L.Trainer(max_epochs=MAX_EPOCHS,callbacks=[checkpoint_callback,early_stop_callback],accelerator=DEVICE,
#                         devices=1,log_every_n_steps=10,limit_train_batches=0.1,limit_val_batches=0.1,max_time={"minutes": 5})
#     trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=validation_loader)
#     return checkpoint_callback,trainer
# checkpoint_callback,trainer=run_fvit_trainer_fit(
#     train_loader,
#     validation_loader,
#      model=model_fvit)

# %%
class FVIT(nn.Module):
    def __init__(self,encoder,decoder):
        super(FVIT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hidden_states=self.encoder(x[0]).last_hidden_state
        # print("Hidden states shape:", hidden_states.shape)
        # print("Additional input shape:", x[1].shape)
        # print("Concatenated input shape:", concatenated_input.shape)
        return self.decoder(hidden_states)
decoder= nn.Sequential(nn.Flatten(),
                       nn.Linear(512*8*8, NUM_CLASSES),
                         nn.Softmax(dim=1)
                   )
# model_fvit=FVIT(model_vitv2,decoder)

# %%
fvit = FVIT(model_vitv2,decoder)
model_fvit = LFVIT(fvit)

# %%
MAX_EPOCHS=50
# timer =  L.pytorch.calalbacks.Timer()
def run_fvit_trainer_fit(train_loader,validation_loader,model):
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='train_loss',
        filename='iec104-{epoch:02d}-{loss}',
        save_top_k=3,
        mode='min',
    )

    # early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")
    # cpb=CustomProgressBar()
    trainer = L.Trainer(max_epochs=MAX_EPOCHS,callbacks=[checkpoint_callback],accelerator=DEVICE,
                        devices=1,log_every_n_steps=10,limit_train_batches=0.1,limit_val_batches=0.1,max_time={"minutes": 5})
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=validation_loader)
    return checkpoint_callback,trainer
checkpoint_callback,trainer=run_fvit_trainer_fit(
    train_loader,
    validation_loader,
     model=model_fvit)

# %%
trainer.test(ckpt_path="best",dataloaders=test_loader)

# %%


# %%
# class FVIT(nn.Module):
#     def __init__(self,encoder,decoder):
#         super(FVIT, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         hidden_states=self.encoder(x[0]).last_hidden_state
#         # print("Hidden states shape:", hidden_states.shape)
#         # print("Additional input shape:", x[1].shape)
#         # print("Concatenated input shape:", concatenated_input.shape)
#         return self.decoder(hidden_states)
decoder= nn.Sequential(nn.Flatten(),
                       nn.Linear(512*8*8, 100),            nn.ReLU(),
                         
            nn.Linear(100, 50),
nn.Softmax(dim=1),
                   )
# model_fvit=FVIT(model_vitv2,decoder)


