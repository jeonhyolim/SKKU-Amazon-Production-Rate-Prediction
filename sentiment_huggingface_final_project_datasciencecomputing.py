import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score

# hugging face
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import XLNetTokenizer, XLNetModel
from transformers import RobertaTokenizer, RobertaModel

class Arg:
    version = 1
    # data
    epochs: int = 10# 10  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_size = 768 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    batch_size: int = 16 #32
    max_post_num = 30
    task_num: int = 1  

class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = "Rate" 
        self.label_names = [0,1,2,3,4] 
        self.num_labels = 5
        self.seed = self.config['random_seed'] 
        self.embed_type = self.config['embed_type'] 

        if self.embed_type == "sentence-transformers":
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        elif self.embed_type == "xlnet-base-cased":
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = XLNetModel.from_pretrained('xlnet-base-cased')
        
        elif self.embed_type == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')

        self.dropout = nn.Dropout(self.config['dropout'])
        self.fc = nn.Linear(self.args.hidden_size, self.num_labels)
        
    def forward(self, input_ids, **kwargs):
    
        outputs_data = self.model(input_ids =input_ids, **kwargs) 
        
        if self.embed_type == "xlnet-base-cased":
            output = outputs_data.last_hidden_state[:,0,:]
        else : 
            output = outputs_data[1]    
        logits = self.fc(self.dropout(output))
        print(logits.shape)
        return logits           

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        
        col_name = 'ReviewDescription' 
        df_train = pd.read_csv('train_dataset.csv')
        df_train['Rate'] = df_train['Rate'].apply(lambda x:x-1) # 0~4로 변경(파이썬 변수특성)
        
        df_test = pd.read_csv('test_dataset.csv')
        df_test['Rate'] = 5
        df_test['Rate'] = df_test['Rate'].apply(lambda x:x-1) # 0~4로 변경(파이썬 변수특성)
        
        #df_train,df_test = train_test_split(df, test_size=0.2, random_state=2023)
        
        df_train = df_train[['ReviewDescription', self.label_cols]]
        df_train=df_train.explode(['ReviewDescription',  self.label_cols])
        
        df_test = df_test[['ReviewDescription', self.label_cols]]
        df_test=df_test.explode(['ReviewDescription', self.label_cols])        
        
        df_train[col_name] = df_train[col_name].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True
            ))
        df_test[col_name] = df_test[col_name].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True
            ))

        self.train_data = TensorDataset(
            torch.tensor(df_train[col_name].tolist(), dtype=torch.long),
            torch.tensor(df_train[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
            torch.tensor(df_test[col_name].tolist(), dtype=torch.long),
            torch.tensor(df_test[self.label_cols].tolist(), dtype=torch.long),
        )

    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        token, labels = batch  
        logits = self(input_ids=token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        self.log("train_loss", loss)

        return {'loss': loss}
            
    def test_step(self, batch, batch_idx):
        token , labels = batch  
        logits = self(input_ids=token) 
        loss = nn.CrossEntropyLoss()(logits, labels)  

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def test_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)
        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        
        print('----y_pred----')
        print(y_pred)
        print('----y_true----')
        print(y_true)
        df_test= pd.DataFrame()
        df_test['id'] = list(range(0,1000))
        df_test['Rate'] = y_pred
        df_test['Rate'] = df_test['Rate'].apply(lambda x:x+1)  # 다시 돌려줌

        # Result Save
        save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
        save_path = f"/home/dsail/hyolim/semester/23-2_semester2/DSC_class_ass/Final_project/_Result/submit/"
        Path(f"{save_path}/pred").mkdir(parents=True, exist_ok=True)

        #pd.DataFrame(val_df).to_csv(f'{save_path}{save_time}.csv',index=False)  
        df_test.to_csv(f'{save_path}pred/{save_time}_pred.csv',index=False)

        return {'loss': _loss} #, 'log': tensorboard_logs}
    
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
        
    model = Model(args,config) 
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    print(":: Start Training ::")
        
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback],
        enable_checkpointing = False,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        gpus=[config['gpu']] if torch.cuda.is_available() else None, 
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader())
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dropout", type=float, default=0.1,help="dropout probablity")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--gpu", type=int, default=1,  help="save fname")
    parser.add_argument("--random_seed", type=int, default=2023)  
    parser.add_argument("--embed_type", type=str, default="roberta-base") 
       
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       

# xlnet: https://huggingface.co/xlnet-base-cased
# roberta: https://huggingface.co/roberta-base
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2