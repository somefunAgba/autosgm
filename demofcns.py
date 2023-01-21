
# Register Functions as a method of an existing Class Object
# clsmethod
def clsmethod(Class): #@save
  def wrapper(obj):
    setattr(Class, obj.__name__,obj)
  return wrapper

#
import math,random,os,copy,time,glob,itertools,shutil,re, json,sys
from scipy.stats import wilcoxon

# terminal_output = open('dev/stdout','w') # linux
terminal_output = open(1,'w') # win
# terminal_error = open('dev/stderr','w')
#
import numpy as np

#
import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch import Tensor
from typing import List, Union, Optional

#
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

    
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

#
import pandas as pd


#
from demofcns import *
from opts.asgm.torchlasgm import PID



def add_to_thisnn(thisNN,worker_seed,setseed,cseedwk) -> None:
  
  @clsmethod(thisNN)
  def reset_parameters(self:thisNN, init_type="kaiming-normal") -> None:
    
    def init_params(m, init_type="kaiming-normal") -> None:

      # - init learnable parameters
      if isinstance(m,(nn.Linear)):
        if init_type == "kaiming-normal":
          nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        elif init_type == "kaiming-uniform":
          nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
        else:
          nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu')) 
        if m.bias is not None: # 0
          nn.init.constant_(m.bias,0)
      elif isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d)):
          nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')          
          if m.bias is not None: # 0
            nn.init.constant_(m.bias,0)
      elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.GroupNorm,nn.LayerNorm)): # 1
        nn.init.constant_(m.weight,1) # set btw 0 and 1
        if m.bias is not None: # 0
          nn.init.constant_(m.bias,0) 
      else:         
          pass
        
    self.apply(fn=init_params)
      
      
  @clsmethod(thisNN)
  def fwdpass_input_to_loss(self:thisNN,x,ytrue):
        # out:[Tensor,List[Tensor],Tensor]:
    # forward pass x -> loss
    
    self.device = "cpu"
    if self.devcnt > 0:
      x = x.to("cuda")
      self.device = "cuda"
      ytrue = ytrue.to(torch.double)
      ytrue = ytrue.to("cuda")
    
    logits = self.forward(x)
    if self.num_heads == 1:
      # convert B x 1 tensor to a list of [B x 1] x 1 element
      # and: ensure same shape with prediction
      if self.outs_class and self.loss_type == "bce":
        ytrue = [ytrue.view(logits[0].shape).to(logits[0].dtype)]
      elif self.outs_class and self.loss_type == "mce":
        ytrue = [ytrue.to(torch.long)] # mce
      else:
        ytrue = [ytrue.to(logits[0].dtype)]
    else:
      # convert B x H tensor to a list of [B x 1] x H elements
      ytrue = (ytrue.T).tolist()
      # ensure same shape with prediction
      for hid in range(self.num_heads):
        if self.outs_class  and self.loss_type == "mce":
          ytrue[hid] = torch.tensor([ytrue[hid]], device=logits[hid].device).T.squeeze().to(torch.long)
        elif self.outs_class  and self.loss_type == "bce":
          ytrue[hid] = torch.tensor([ytrue[hid]], dtype=logits[0].dtype, device=logits[hid].device).T.squeeze().view(logits[hid].shape)
          #.flatten().tolist()
        else:
          ytrue[hid] = torch.tensor([ytrue[hid]], dtype=torch.float32, device=logits[hid].device).T.squeeze()
    

    if self.loss_type == "mse": 
      yhat = self.outhead(logits)
    
      f = 0
      # add up losses for each head
      for hid in range(self.num_heads):
        yhat_hid = yhat[hid]
        f_hid = tf.mse_loss(yhat_hid,ytrue[hid])
        f = f + torch.div(f_hid,self.num_heads)
        
    elif self.loss_type == "mce" and not self.outs_class:
      yhat = self.outhead(logits)
      
      f = 0
      # add up losses for each head
      for hid in range(self.num_heads):
        yhat_hid = yhat[hid]
        f_hid = tf.cross_entropy(yhat_hid,ytrue[hid])
        f = f + torch.div(f_hid,self.num_heads) # f.append(f_hid)
      
    elif self.loss_type == "bce" and not self.outs_class:
      yhat = self.outhead(logits)
      
      f = 0
      # add up losses for each head
      for hid in range(self.num_heads):
        yhat_hid = yhat[hid]
        f_hid = tf.binary_cross_entropy(yhat_hid,ytrue[hid])
        f = f + torch.div(f_hid,self.num_heads)
      
    elif self.loss_type == "mce" and self.outs_class:
      yhat = []
      
      f = 0
      # add up losses for each head
      for hid in range(self.num_heads):
        f_hid = tf.cross_entropy(logits[hid],ytrue[hid])
        f = f + torch.div(f_hid,self.num_heads) # f.append(f_hid/self.num_heads)
        
        yhat_hid = logits[hid].argmax(1) 
        yhat.append(yhat_hid)
        # yhat = self.outhead(logits)

    elif self.loss_type == "bce" and self.outs_class:
      yhat = [] 
      # yhat = self.outhead(logits)
      
      f = 0
      # add up losses for each head
      for hid in range(self.num_heads):
        f_hid = tf.binary_cross_entropy_with_logits(logits[hid],ytrue[hid])
        f = f + torch.div(f_hid,self.num_heads)
        
        yhat_hid = torch.where(logits[hid] > 0, 1., 0.) #
        yhat.append(yhat_hid)
        # yhat[hid] = torch.where(yhat[hid] > 0.5, 1., 0.) #
        
        
    # compute: number of correct predictions
    correct = 0
    if self.outs_class:
      for hid in range(self.num_heads):
        correct = correct + torch.div((yhat[hid]==ytrue[hid]).type(torch.float).sum().item(),self.num_heads)
    else:
      correct = None
    
    return f, yhat, correct

  @torch.no_grad()
  @clsmethod(thisNN)
  def infer(self:thisNN,x):
    if self.devcnt > 0:
      x = x.to("cuda")
      
    logits = self.forward(x)
    yhat = self.outhead(logits)
    
    for hid in range(self.num_heads):
      if self.outs_class and self.loss_type =="mce":
        yhat[hid] = yhat[hid].argmax(1).squeeze(-1)
      if self.outs_class and self.loss_type =="bce":
        yhat[hid] = torch.where(yhat[hid] > 0.5, 1., 0.).squeeze(-1)
        
    return torch.stack(yhat).mT
    

  @clsmethod(thisNN)
  def learn_onestep(self:thisNN,x,ytrue):

    loss, yhat, corr = self.fwdpass_input_to_loss(x,ytrue)
    
    self.sgm.zero_grad(set_to_none=True)
    # for param in self.parameters():
    #   param.grad = None 
    loss.backward()
    self.sgm.step()
    
    return loss, yhat, corr


  @clsmethod(thisNN)
  def eval_loop(self:thisNN,eval_dataloader,cfgs=None, eval_name="Test"):
    data_size = len(eval_dataloader.dataset)
    num_batches = len(eval_dataloader)
    
    self.eval() # configure model in eval. mode.
    
    walltime = time.time()
    predictions = []
    for hid in range(self.num_heads):
      predictions.append([])
    
    eval_loss, correct = 0., 0.
    for batch, (data_in_batch,data_truth_batch) in enumerate(eval_dataloader):
      # a single batch
      loss, pred_out, correct_one_batch = self.fwdpass_input_to_loss(data_in_batch,data_truth_batch)
      
      for hid in range(self.num_heads):
        try:
          predictions[hid].append(list(itertools.chain.from_iterable(pred_out[hid].tolist())))
        except:
          predictions[hid].append((pred_out[hid].tolist()))
      
      # metrics
      eval_loss += loss.item() 
      if self.outs_class:
        correct += correct_one_batch
      else:
        correct = None
    
    # wall-time for a single training run 
    walltime = (time.time() - walltime)/60   
    # combine predictions in one data structure
    for hid in range(self.num_heads):
      predictions[hid] = list(itertools.chain.from_iterable(predictions[hid]))  
    
    eval_loss = eval_loss/num_batches
    if self.outs_class:
      # classifier accuracy
      eval_accuracy = 100*correct.item()/data_size
    else:
      # mean accuracy
      eval_accuracy = 1 - eval_loss
    
    # logs
    print(f"{eval_name}: [ Avg Loss: {eval_loss:>0.4f}, Accuracy: {(eval_accuracy):>0.2f}% ]")
    print(f"Elapsed Inf. time: {walltime:.2f}-mins.\n ")
    
    
    return eval_loss, eval_accuracy, predictions
    

  @clsmethod(thisNN)
  def train_loop(self:thisNN, train_dataloader, test_dataloader, cfgs=None, eval_name="Test"):
    epochs = cfgs['epochs']
    PathStr = cfgs['pathstr']
    storedir = cfgs['storedir']
    mdl_name_dir = cfgs['mdl_name_dir']
    
    data_size = len(train_dataloader.dataset)
    # number of learning steps performed over the dataset size in one epoch.
    # steps_per_each_epoch = data size / batch size 
    # i.e: number of batches for any given batch size, and data size
    steps_per_epoch = len(train_dataloader)
    num_batches = steps_per_epoch
    
    # reset or init parameters before a single training run
    self.reset_parameters()
    
    # assign parameters to SGM
    self.sgm = PID(self.parameters(), steps_per_epoch=steps_per_epoch, ss_init=cfgs["ss_init"], eps_ss=cfgs["eps_ss"], weight_decay=cfgs["weight_decay"])
    
    # a single training run (epoch '1' -> epoch 'epochs')
    best_loss, best_acc= 100., -0.,
    # epoch metrics list
    train_losses, test_losses, train_accs, test_accs = [],[],[],[]
    for t in range(epochs):
      self.train() # configure model in train mode.
      train_loss, correct = 0., 0.
      
      walltime = time.time()
      # a single epoch
      for k, (data_in_batch,data_truth_batch) in enumerate(train_dataloader):
        # k = a single step out of steps[_per_epoch] equivalent to
        # k = a single batch out of num_batches
        loss, pred_out, correct_one_batch = self.learn_onestep(data_in_batch,data_truth_batch)
        
        # metrics
        train_loss += loss.item() 
        if self.outs_class:
          correct += correct_one_batch
        else:
          correct = None
          
      # wall-time for a single training epoch
      walltime = (time.time() - walltime)/60   
      
      train_loss = train_loss/num_batches
      if self.outs_class:
        # classifier accuracy
        train_accuracy = 100*correct.item()/data_size
      else:
        # mean accuracy
        train_accuracy = 1 - train_loss
      
      # logs.
      # train logs.
      print(f"{t+1}: Elapsed Train time: {walltime:.2f}-mins.") 
      print(f"Batches/Steps/Iterations per Epoch: {num_batches:>5d}")
      print(f"Train:\t[ Avg Loss: {train_loss:>0.4f}, Accuracy: {(train_accuracy):>0.2f}% ]", end=' || ')
      
      # test       
      test_loss, test_accuracy, test_preds = self.eval_loop(test_dataloader,cfgs=None, eval_name=eval_name)
      
      # append loss and acc metrics, for each epoch, to list
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      train_accs.append(train_accuracy)
      test_accs.append(test_accuracy)
      
      # save better model of the test data at each epoch (Validation)
      # the last saved model is the best for a single training run
      if (best_acc < test_accuracy):  # or (oloss > loss):
        
        path_to_saved_mdl = f"{storedir}/{mdl_name_dir}/stores/mdls"
        os.makedirs(path_to_saved_mdl, exist_ok=True)
        
        # mets_path = f"{path_to_saved_mdl}/mets_{PathStr}.pt"
        
        thispath = f"{path_to_saved_mdl}/{PathStr}.pt"
        
        best_acc, best_loss = test_accuracy, test_loss
        
        # - saving checkpoints
        torch.save({
          'model': self,
          'model_state_dict': self.state_dict(),
          'optimizer_state_dict': self.sgm.state_dict(),
          'train_loss': train_loss,
          'eval_loss': test_loss,
          'train_acc': train_accuracy,
          'eval_acc': test_accuracy,
          'test_preds':test_preds,
          'epoch': t+1,
          'num_batches': num_batches
        },thispath)
        
    # Load saved model info, after each single run:
    # thispath = f"{storedir}/{mdl_name_dir}/stores/mdls/{PathStr}.pt"
    self.chkpt = torch.load(thispath)

    
    # clear optimizer state, after a single training run 
    self.sgm.state.clear()
    
    
    return train_losses, train_accs, test_losses, test_accs
    
    
    
  @clsmethod(thisNN)
  def runs(self:thisNN, train_data, test_data, cfgs=None, eval_name="Test"):
    
    batch_size = cfgs["batch-size"]
    num_workers = cfgs["num-workers"] 
    epochs = cfgs['epochs']
    runs = cfgs['runs']
    PathStr = cfgs['pathstr']
    storedir = cfgs['storedir']
    mdl_name_dir = cfgs['mdl_name_dir']
    
    # metrics: lists
    train_losses_list_per_run, dev_losses_list_per_run, train_accs_list_per_run,dev_accs_list_per_run, best_preds_list_per_run = [],[],[],[],[]

    # logic to vary seed for each run
    var_seed = bool(cfgs['var_seed'])
    
    # debug
    print(f"Seed: {torch.initial_seed()}")
    
    # Training Runs for this modeling experiment: 
    # for each run, use the defined train and test loop
    for rid in range(runs):
      
      # vary seed
      # ensure seed is set to desired value before each run.
      setseed(rid) if var_seed else setseed()
      wk_seed = worker_seed
      cseedwk.inc = 0
      if var_seed:
        wk_seed = worker_seed + rid
        cseedwk.inc = rid  

      # fix seed for batch generation
      gen_seed = torch.Generator()
      gen_seed.manual_seed(wk_seed)
      
      # load dataloader for this modeling experiment
      train_dataloader = DataLoader(train_data,batch_size=batch_size, num_workers=num_workers, shuffle=True,worker_init_fn=cseedwk.seed_worker,generator=gen_seed)
      #
      test_dataloader = DataLoader(test_data,batch_size=batch_size,num_workers=num_workers,worker_init_fn=cseedwk.seed_worker,generator=gen_seed)
      
      # train for defined number of epochs
      train_loss_list_per_epoch, train_accuracy_list_per_epoch, dev_loss_list_per_epoch, dev_accuracy_list_per_epoch = self.train_loop(train_dataloader, test_dataloader, cfgs, eval_name)
      
      # append model's best test-predictions, for each run, to list
      best_preds_list_per_run.append(self.chkpt['test_preds'])
      # append loss and acc metrics, for each run, to list
      train_losses_list_per_run.append(train_loss_list_per_epoch)
      train_accs_list_per_run.append(train_accuracy_list_per_epoch)
      dev_losses_list_per_run.append(dev_loss_list_per_epoch)
      dev_accs_list_per_run.append(dev_accuracy_list_per_epoch)
      
      bid = self.chkpt['epoch']-1
      # - plot and individual test-train loss Metrics of each run
      PATHplots = f"{storedir}/stores/plots"
      os.makedirs(PATHplots, exist_ok=True)
      PATHplots = f"{PATHplots}/plots_{PathStr}"
      
      # dashplots.traintest(train_loss_list_per_epoch, train_accuracy_list_per_epoch, dev_loss_list_per_epoch, dev_accuracy_list_per_epoch, epochs, self.chkpt['num_batches'],figname=PATHplots, live=False)
      
      # debug
      strlog = f"Done! Run: {rid+1} : [Best {eval_name} @ epoch {self.chkpt['epoch']}] || [ Avg loss: {self.chkpt['eval_loss']:>7f}, Accuracy: {(self.chkpt['eval_acc']):>0.2f}% ]"    
      txt = "*"
      print(f"{txt * len(strlog)}")
      print(strlog)
      print(f"{txt * len(strlog)}\n")
    
    # ********************  
    # post-run operations
    
    # get test-set ground-truth
    try:
      ground_truth = test_dataloader.dataset.targets.tolist()
    except:
      try:
        ground_truth = test_dataloader.dataset.targets
      except:
          try:
            ground_truth = test_dataloader.dataset.labels.tolist()
          except:
              try:
                ground_truth = test_dataloader.dataset.labels
              except:
                  pass
    
    # append truest prediction to best_preds list
    if self.num_heads == 1:
      best_preds_list_per_run.append([ground_truth])
    else:
      best_preds_list_per_run.append(ground_truth)
    
    # - save best test predictions for the just completed runs
    PATHpreds = f"{storedir}/{mdl_name_dir}/stores/preds"
    os.makedirs(PATHpreds, exist_ok=True)
    PATHpreds = f"{PATHpreds}/preds_{PathStr}"
    df = pd.DataFrame(best_preds_list_per_run)
    df = df.T
    df.to_csv(PATHpreds+".csv")
    # print(df.head()) print(df.tail())
    
    # - save test loss and acc. metrics for the just completed runs
    PATHpreds = f"{storedir}/{mdl_name_dir}/stores/acc"
    os.makedirs(PATHpreds, exist_ok=True)
    
    df = pd.DataFrame(train_losses_list_per_run)
    df = df.T
    thisPATH = f"{PATHpreds}/trainloss_{PathStr}"
    df.to_csv(thisPATH+".csv")
    
    df = pd.DataFrame(dev_losses_list_per_run)
    df = df.T
    thisPATH = f"{PATHpreds}/devloss_{PathStr}"
    df.to_csv(thisPATH+".csv")
    
    df = pd.DataFrame(dev_accs_list_per_run)
    df = df.T
    thisPATH = f"{PATHpreds}/devacc_{PathStr}"
    df.to_csv(thisPATH+".csv")
    
    df = pd.DataFrame(train_accs_list_per_run)
    df = df.T
    thisPATH = f"{PATHpreds}/trainacc_{PathStr}"
    df.to_csv(thisPATH+".csv")
    # to_csv's index is true by default, can change to false to remove row ids
    
    
    # - Write current cfg to the stores folder for book-keeping
    run_cfgs = copy.deepcopy(cfgs)
    run_cfgs['eff_test_accuracy'] = 'None'
    run_cfgs['med_stat'] = 'None'
    run_cfgs['pval'] = 'None'
    run_cfgs["device"] = self.device
      
    PATHruncfg = f"{storedir}/{mdl_name_dir}/stores/exp_cfg"
    os.makedirs(PATHruncfg, exist_ok=True)
      
    thiscfgPATH = f"{PATHruncfg}/cfgs_{PathStr}.json"

    # Reproducibility metrics
    if runs > 1 and self.outs_class:
      # - Compute Best Prediction Consistency across Runs
      pdiff_list= []
      actual_pdiff_list=[]
      
      lastid = len(best_preds_list_per_run)-1 # = runs
      assert runs == lastid
      # total num of data points evaluated
      total_datapts = len(ground_truth)
      
      # Total difference between Predictions across several runs
      for iid in range(runs-1):
        for jid in range(iid+1, runs):
          act_pred_diff = 0
          for hid in range(self.num_heads):
            act_pred_diff = act_pred_diff + sum(np.array(best_preds_list_per_run[iid][hid]) != np.array(best_preds_list_per_run[jid][hid])) 

          pred_diff = act_pred_diff/(self.num_heads*total_datapts)
            
          actual_pdiff_list.append(act_pred_diff)
          pdiff_list.append(pred_diff)
          
      PATHpd = f"{storedir}/{mdl_name_dir}/stores/pdiff"
      os.makedirs(PATHpd, exist_ok=True)
      
      # Compute Effective Test-Accuracy:
      # mean accuracy - mean pred.difference
      # eff_test_acc = (np.mean(np.array(dev_accs_list_per_run))-np.mean(np.array(actual_pdiff_list)))/total_datapts  
      avg_test_acc = (np.mean(np.max(np.array(dev_accs_list_per_run),1)))
      avg_pred_diff = ((np.mean(np.array(pdiff_list))))  
      eff_test_acc = avg_test_acc - avg_pred_diff
      
      # - Save Pdiff  
      df = pd.DataFrame(pdiff_list)
      thisPATH = f"{PATHpd}/pdiff_{PathStr}"
      df.to_csv(thisPATH+".csv")
      
      df = pd.DataFrame(actual_pdiff_list)
      thisPATH = f"{PATHpd}/actpdiff_{PathStr}"
      df.to_csv(thisPATH+".csv")
      
      # The Wilcoxon T-test. Given n independent samples (xi, yi) from a bivariate distribution (i.e. paired samples), it computes differences di = xi - yi. (OR. Skip this: by supplying with the paired differences, di). One assumption of the test is that the differences are symmetric. 
      # The two-sided test has the null hypothesis that the median of the differences is zero against the alternative that it is different from zero.
      try:
        med_stat, pval = wilcoxon(pdiff_list)
      except ValueError:
        # ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        med_stat, pval = -1, 0.5 # indicative of when the predictions are the same,
      # if pval isn't 0.5, we would reject the null hypothesis at a confidence level of 5%, concluding that the pred. diff across runs is significant.
      
      # - Write current cfg to the stores folder for book-keeping
      run_cfgs['eff_test_accuracy'] = eff_test_acc
      try:
        run_cfgs['med_stat'] = med_stat[0]
        run_cfgs['pval'] = pval[0]
      except:
        run_cfgs['med_stat'] = med_stat
        run_cfgs['pval'] = pval
      
      strlog = f"Effective Test Accuracy = {eff_test_acc:.2f}%, pval={pval:.4f}"
      print(f"{txt * len(strlog)}")    
      print(f"Average Prediction Difference = {avg_pred_diff:.2f}%")
      print(strlog)
      print(f"{txt * len(strlog)}\n")   
      
    # save
    with open(thiscfgPATH, 'w') as cfglist:
        json.dump(run_cfgs, cfglist)
      
      
cmd_exists = lambda x: shutil.which(x) is not None
if cmd_exists('latex'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif", # serif
        "font.sans-serif": "Helvetica", # Times, Palatino, Computer Modern Roman
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
    })


def plotter_v1(cfgs,run_idx=1):

  try:
    saved_cfgs_path = f"{storedir}/{mdl_name_dir}/stores/exp_cfg/cfgs_{pathstr}.json"
    with open(saved_cfgs_path, 'r') as cfglist:
            scfgs = json.load(cfglist)
  except:
    scfgs = cfgs
          
  pathstr = scfgs['pathstr']
  storedir = scfgs['storedir']
  mdl_name_dir = scfgs['mdl_name_dir']

  # print(df.head()) print(df.tail())

  # os.makedirs(PATHpreds, exist_ok=True)

  # which training run?

  # train_losses_ list_per_run train_accs_list_per_run
  path_train_loss = f"{storedir}/{mdl_name_dir}/stores/acc/trainloss_{pathstr}"
  path_train_acc= f"{storedir}/{mdl_name_dir}/stores/acc/trainacc_{pathstr}"
  path_dev_loss = f"{storedir}/{mdl_name_dir}/stores/acc/devloss_{pathstr}"
  path_dev_acc= f"{storedir}/{mdl_name_dir}/stores/acc/devacc_{pathstr}"

  dfl = pd.read_csv(path_train_loss+".csv")
  dfa = pd.read_csv(path_train_acc+".csv")

  # access the data
  train_loss_per_run = dfl.iloc[0:,run_idx]
  train_acc_per_run = dfa.iloc[0:,run_idx]

  dfl = pd.read_csv(path_dev_loss+".csv")
  dfa = pd.read_csv(path_dev_acc+".csv")

  # access the data
  dev_loss_per_run = dfl.iloc[0:,run_idx]
  dev_acc_per_run = dfa.iloc[0:,run_idx]


  #inputs:
  epochs_t = np.arange(cfgs['epochs'])
  trainlosses = np.array(train_loss_per_run)
  accs = []


  # Line plot ----------
  figsz = (3*1.5, 2*1.5)
  fig1 = plt.figure(figsize=figsz,tight_layout=True)
  gridpanel  = gridspec.GridSpec(1,1,)
  ax1 = plt.subplot(gridpanel[0])

  # Plot training and validation curves
  # fig, ax1 = plt.subplots(figsize=figsz)
  color = 'tab:red'
  l1 = ax1.plot(epochs_t, train_loss_per_run.values, linestyle='dashed', c=color,alpha=0.1, label=r"$\mathrm{\mathsf{loss\/(train)}}$")
  l2 = ax1.plot(epochs_t, dev_loss_per_run.values, c=color,
  alpha=0.5, label=r'$\mathrm{\mathsf{loss\/(test)}}$')
  # marker='.', markersize=3, markevery=10
  ax1.set_xlabel(r"$\mathrm{\mathsf{epochs}}$")
  ax1.set_ylabel(r'$\mathrm{\mathsf{loss}}$', c=color)
  # ax1.tick_params(axis='y', labelcolor=color)
  # ax1.set_ylim(-0.01,3)

  color = 'tab:blue'
  ax12 = ax1.twinx()
  l3 = ax12.plot(epochs_t, train_acc_per_run.values, linestyle='dashed', c=color,alpha=0.1, label=r"$\mathrm{\mathsf{accuracy\/(train)}}$")
  l4 = ax12.plot(epochs_t, dev_acc_per_run.values, c=color,
  alpha=0.5, label=r"$\mathrm{\mathsf{accuracy\/(test)}}$")
  ax12.set_ylabel(r"$\mathrm{\mathsf{accuracy}}$", c=color)
  # ax1.tick_params(axis='y', labelcolor=color)
  # ax1.set_ylim(-0.01,3)

  # ax12.axhline(y=scfgs['eff_test_accuracy'],c=color,alpha=0.8)
  # marker='.', markersize=3, markevery=10
  ax1.legend(handles=l1+l2+l3+l4, loc=(0.45,0.45), fontsize=8)
  fig1.savefig(f"{storedir}/{mdl_name_dir}/stores/train_test_curves.png", dpi=300)

  # Line plot ----------
  if cfgs["runs"] > 1:
      path_preds= f"{storedir}/{mdl_name_dir}/stores/pdiff/pdiff_{pathstr}"
      dfp = pd.read_csv(path_preds+".csv")
      dev_preds_per_run = dfp.iloc[0:,1]

      figsz = (3*1.5, 2*1.5)
      fig2 = plt.figure(figsize=figsz,tight_layout=True)
      gridpanel  = gridspec.GridSpec(1,1,)
      ax2 = plt.subplot(gridpanel[0])

      color = 'goldenrod'
      ax2.plot(dev_preds_per_run.values,'.', c=color, 
      alpha=0.9, markersize=3)
      ax2.set_xlabel(r"$\mathrm{\mathsf{comparisons}}$")
      ax2.set_ylabel(r"$\mathrm{\mathsf{prediction\/difference}}$")
      # ax1.set_ylim(-0.01,3)
      # ax2.legend(loc="best", fontsize=8)

      plt.show()
      fig2.savefig(f"{storedir}/{mdl_name_dir}/stores/pred_diffs_runs.png", dpi=300)