import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import time
import os 
from PIL import Image

def plot_Train_dataset(train, params, hparams, name):
    
    # plot train dataset distribution 
    # total distribution of samples
    # distribution of positive iteractions
    # distribution of negative sampling

    ds_train =  pd.DataFrame(train, columns=['users','items','ratings'])
  
    drec1 = ds_train[ds_train['ratings']==0]
    drec2 = ds_train[ds_train['ratings']==1]
  
    drec = ds_train.groupby(['items'])['items'].agg(cuenta='count').sort_values(['cuenta'], ascending=False).reset_index()
    drec1 = drec1.groupby(['items'])['items'].agg(cuenta='count').sort_values(['cuenta'], ascending=False).reset_index()
    drec2 = drec2.groupby(['items'])['items'].agg(cuenta='count').sort_values(['cuenta'], ascending=False).reset_index()
 
    num_items = drec.shape[0]
    num_items_NEG = drec1.shape[0]
    num_items_POS = drec2.shape[0]        
    num_samples = ds_train.shape[0]

    fig  = px.bar(drec, x='items', y='cuenta', color='cuenta', title=name+".Train dataset with "+str(num_items) + " different items and "+str(num_samples)+" samples<br>params: num_neg="+str(params['num_neg'])+" rating="+params['leave_one_out']+" batch_size="+str(hparams['batch_size']) + " hidden=" + str(hparams['hidden_size']))
    fig1 = px.bar(drec1, x='items', y='cuenta', color='cuenta', title=name+". Negative samples ")
    fig2 = px.bar(drec2, x='items', y='cuenta', color='cuenta', title=name+". Positive samples ")

    fig.update_xaxes(type='category')  
    fig1.update_xaxes(type='category') 
    fig2.update_xaxes(type='category') 
   
    fig.write_image(os.getcwd() + "/logs/" + "train_TOT_ds_num_neg_"+str(params['num_neg'])+"_rating_"+params['leave_one_out']+"_batch_size_"+str(hparams['batch_size']) + "_hidden_" + str(hparams['hidden_size']) + ".png")
    time.sleep(2)
    fig1.write_image(os.getcwd() + "/logs/" + "train_NEG_ds_num_neg_"+str(params['num_neg'])+"_rating_"+params['leave_one_out']+"_batch_size_"+str(hparams['batch_size']) + "_hidden_" + str(hparams['hidden_size']) + ".png")
    time.sleep(2)    
    fig2.write_image(os.getcwd() + "/logs/" + "train_POS_ds_num_neg_"+str(params['num_neg'])+"_rating_"+params['leave_one_out']+"_batch_size_"+str(hparams['batch_size']) + "_hidden_" + str(hparams['hidden_size']) + ".png")


def plot_Reco_vs_POP(listREC, listPOP, name, epoch, num_epochs, model):
    # PLOT ONLY LAST EPOCH
    if (int(epoch) >= (num_epochs-1)):
        # put in a row
        lrecommended = np.hstack(listREC)
        drec =  pd.DataFrame(lrecommended, columns=['itemrec'])
        lPopular = np.hstack(listPOP)
        dpop =  pd.DataFrame(lrecommended, columns=['itempop'])
    
        drec1 = drec.groupby(['itemrec'])['itemrec'].agg(cuenta='count').sort_values(['cuenta'], ascending=False).reset_index()
  
        fig = px.bar(drec1, x='itemrec', y='cuenta', color='cuenta', title="EPOCH:"+ epoch +" - Recommended items - number of different items="+str(len(set(lrecommended)))+" <br>Items also in popularity list="+str(len(set(set(lrecommended) & set(lPopular)))) + " - model=" + model)
        fig.update_xaxes(type='category')  
    
        fig.write_image(os.getcwd() + "/logs/" + name)
        time.sleep(2)

def show_generated_plots():
    logs_dir = os.getcwd() + "/logs/"
    logs_content = os.listdir(logs_dir)
    for filename in logs_content:
        if not ".png" in filename:
            logs_content.remove(filename)
        else:
            path = str(os.getcwd() + "/logs/" + filename)
            im = Image.open(rf"{path}")
            im.show()