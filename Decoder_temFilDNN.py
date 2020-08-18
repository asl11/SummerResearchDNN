import sys
numlayers = 2#int(sys.argv[1])
KinV = 0#int(sys.argv[1])
KinV2 = 4#int(sys.argv[2])
saveName = "test" #str(sys.argv[3])
Neural_ensemble_partStart =0# int(sys.argv[1])#0
Neural_ensemble_partEnd = 1#int(sys.argv[2])#0
gpuNum = str(0)#sys.argv[3]#str(0)#
newData = 0 #Previously was initialized, so we can leave it at 0 now
# coding: utf-8
from scipy import io
import numpy as np
# # All decoders (except KF and ensemble) run on full datasets
zeroCenterOutput=False
normalizeOutputByStd=False
# ## User Options
#num_units=10
#n_epoch=100
#batch_size=32
#final_act='relu'
#KinV = 4

save_folder='/home/hanlin/Neural_Decoding/Results/'
load_folder='/home/hanlin/Neural_Decoding/package0403/'
dataset='xie'
run_wf=0 #Wiener FilterDataPartitionID
run_wc=0 #Wiener Cascade
run_svr=0 #Support vector regression
run_xgb=0 #XGBoost
run_dnn=1 #Feedforward (dense) neural network
run_dnn_ISA=0
run_rnn=0 #Recurrent neural network
run_gru=0 #Gated recurrent units
run_lstm=0 #Long short term memory network
run_lstm_sim=0
# Define what folder you're saving to
fold_num = 3
for Neural_ensemble_part in range(Neural_ensemble_partStart,Neural_ensemble_partEnd):
    print('Neural_ensemble_part %d' ,Neural_ensemble_part)
    if dataset=='xie':
        RawDatafileName = 'Latent12_FR30_Score30'
        # OneBest49Trash
        # 'sortWorstToBest'
        filepath = load_folder + RawDatafileName +'.mat'
        f = io.loadmat(filepath)

        vels_binned =  f['Kin']
        vels_binned = vels_binned[:,KinV:KinV2] #2.1
        # decode y-vel for 7th use tanh, zero center output
        # filepath = load_folder+'most_xcorr_FR_100_50_3v7.mat'
        # f = io.loadmat(filepath)
        if newData:
            neural_data = f['FR']

            #neural_data = neural_data[:1000,:]
        #neural_data = neural_data[:,Neural_ensemble_part*60:min((Neural_ensemble_part+1)*60,2984)]


    if dataset=='m1' or dataset=='xie':
        bins_before=16 #How many bins of neural data prior to the output are used for decoding
        bins_current=1 #Whether to use concurrent time bin of neural data
        bins_after=8 #How many bins of neural data after (and including) the output are used for decoding


    valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],
                     [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
    testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],
                       [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
    #Note that the training set is not aways contiguous. For example, in the second fold, the training set has 0-10% and 30-100%.
    #In that example, we enter of list of lists: [[0,.1],[.3,1]]
    training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
                        [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]



    # In[1]:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuNum  # use id from $ nvidia-smi
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement =True
    sess = tf.Session(config=config)
    set_session(sess)

    from keras import backend as K
    import gc
    K.clear_session()
    gc.collect()


    from scipy import stats
    import pickle
    import time
    import sys
    import h5py

    #Add the main folder to the path, so we have access to the files there.
    #Note that if your working directory is not the Paper_code folder, you may need to manually specify the path to the main folder. For example: sys.path.append('/home/jglaser/GitProj/Neural_Decoding')
    sys.path.append('/home/hanlin/Neural_Decoding/Neural_Decoding')
    sys.path.append('/home/hanlin/Neural_Decoding/package0403')
    sys.path.append('.')
    sys.path.append('/home/hanlin/Neural_Decoding')
    sys.path.append('/home/hanlin/Neural_Decoding/package0403/decoders.py')


    #Import function to get the covariate matrix that includes spike history from previous bins
    from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

    #Import metrics
    from Neural_Decoding.metrics import get_R2
    from Neural_Decoding.metrics import get_rho

    #Import decoder functions
    sys.path.append('/home/hanlin/Neural_Decoding/package0403')
    sys.path.append('.')
    sys.path.append('/home/hanlin/Neural_Decoding/package0403/decoders.py')
    from Neural_Decoding.decoders import WienerCascadeDecoder
    from Neural_Decoding.decoders import WienerFilterDecoder
    from Neural_Decoding.decoders import DenseNNDecoder
    from package0403.decoders import DenseNNRegression_Hanlin, DenseNNRegression_Hanlin_IntSuperAdd
    from Neural_Decoding.decoders import SimpleRNNDecoder
    from Neural_Decoding.decoders import GRUDecoder
    from Neural_Decoding.decoders import LSTMDecoder
    from Neural_Decoding.decoders import XGBoostDecoder
    from Neural_Decoding.decoders import SVRDecoder
    from Neural_Decoding.decoders import LSTMClassification
    from package0403.decoders import LSTMClassification_sim

    #Import Bayesian Optimization package
    from bayes_opt import BayesianOptimization


    # In[6]:


    #Turn off deprecation warnings

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    # ## 2. Load Data
    #
    # The data that we load is in the format described below. We have another example script, "Example_format_data" that may be helpful towards putting the data in this format.
    #
    # Neural data should be a matrix of size "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
    #
    # The output you are decoding should be a matrix of size "number of time bins" x "number of features you are decoding"

    # In[7]:



        # filepath = load_folder+'Kin_100_50_v7.mat'
        # f = io.loadmat(filepath)
        # vels_binned = f['Kin']
        # vels_binned = vels_binned[:, 4:5]
        # filepath = load_folder + 'most_xcorr_FR_100_50_3v7.mat'
        # f = io.loadmat(filepath)
        # neural_data = f['mostXFR']
        #     neural_data,vels_binned=pickle.load(f,encoding='latin1')

    # ## 3. Preprocess Data

    # ### 3A. User Inputs
    # The user can define what time period to use spikes from (with respect to the output).

    # In[8]:


    if dataset=='s1':
        bins_before=6 #How many bins of neural data prior to the output are used for decoding
        bins_current=1 #Whether to use concurrent time bin of neural data
        bins_after=6 #How many bins of neural data after (and including) the output are used for decoding



    if dataset=='hc':
        bins_before=4 #How many bins of neural data prior to the output are used for decoding
        bins_current=1 #Whether to use concurrent time bin of neural data
        bins_after=5 #How many bins of neural data after (and including) the output are used for decoding


    # ### 3B. Format Covariates

    # #### Format Input Covariates

    # In[9]:


    #Remove neurons with too few spikes in HC dataset



    # In[10]:


    # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
    # Function to get the covariate matrix that includes spike history from previous bins
    if newData:
        X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)
        X = np.single(np.asarray(X))

    # Format for Wiener Filter, Wiener Cascade, SVR, XGBoost, and Dense Neural Network
    #Put in "flat" format, so each "neuron / time" is a single feature
    #X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))


    # #### Format Output Covariates

    # In[11]:


    #Set decoding output
    if dataset=='s1' or dataset=='m1' or dataset=='xie':
        y=vels_binned





    num_folds=len(valid_range_all) #Number of cross validation folds


    # ## 4. Run CV

    # **Initialize lists of results**

    # In[15]:


    #R2 values
    mean_r2_wf=np.empty(num_folds)
    mean_r2_wc=np.empty(num_folds)
    mean_r2_xgb=np.empty(num_folds)
    mean_r2_svr=np.empty(num_folds)
    mean_r2_dnn=np.empty(num_folds)
    mean_r2_rnn=np.empty(num_folds)
    mean_r2_gru=np.empty(num_folds)
    mean_r2_lstm=np.empty(num_folds)

    #Actual data
    y_test_all=[]
    y_train_all=[]
    y_valid_all=[]

    #Test predictions
    y_pred_wf_all=[]
    y_pred_wc_all=[]
    y_pred_xgb_all=[]
    y_pred_dnn_all=[]
    y_pred_rnn_all=[]
    y_pred_gru_all=[]
    y_pred_lstm_all=[]
    y_pred_svr_all=[]

    #Training predictions
    y_train_pred_wf_all=[]
    y_train_pred_wc_all=[]
    y_train_pred_xgb_all=[]
    y_train_pred_dnn_all=[]
    y_train_pred_rnn_all=[]
    y_train_pred_gru_all=[]
    y_train_pred_lstm_all=[]
    y_train_pred_svr_all=[]

    #Validation predictions
    y_valid_pred_wf_all=[]
    y_valid_pred_wc_all=[]
    y_valid_pred_xgb_all=[]
    y_valid_pred_dnn_all=[]
    y_valid_pred_rnn_all=[]
    y_valid_pred_gru_all=[]
    y_valid_pred_lstm_all=[]
    y_valid_pred_svr_all=[]


    # **In the following section, we**
    # 1. Loop across folds
    # 2. Extract the training/validation/testing data
    # 3. Preprocess the data
    # 4. Run the individual decoders (whichever have been specified in user options). This includes the hyperparameter optimization
    # 5. Save the results
    #
    # Note that the Wiener Filter, Wiener Cascade, and XGBoost decoders are commented most fully. So look at those for the best understanding.

    # In[16]:


    t1=time.time() #If I want to keep track of how much time has elapsed

    num_examples=y.shape[0] #number of examples (rows in the X matrix)


    for i in [fold_num]: #Loop through the folds

        ######### SPLIT DATA INTO TRAINING/TESTING/VALIDATION #########

        #Note that all sets have a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
        #This makes it so that the different sets don't include overlapping neural data

        #Get testing set for this fold
        testing_range=testing_range_all[i]
        testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)

        #Get validation set for this fold
        valid_range=valid_range_all[i]
        valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

        #Get training set for this fold.
        #Note this needs to take into account a non-contiguous training set (see section 3C)
        training_ranges=training_range_all[i]
        for j in range(len(training_ranges)): #Go through different separated portions of the training set
            training_range=training_ranges[j]
            if j==0: #If it's the first portion of the training set, make it the training set
                training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
            if j==1: #If it's the second portion of the training set, concatentate it to the first
                training_set_temp=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
                training_set=np.concatenate((training_set,training_set_temp),axis=0)

        #Get training data
        if newData:
            X_train=X[training_set,:,:]
            X_test = X[testing_set, :, :]
            X_valid = X[valid_set, :, :]
            X = None

        y_train=y[training_set,:]

        #Get testing data


        y_test=y[testing_set,:]

        #Get validation data


        y_valid=y[valid_set,:]


        ##### PREPROCESS DATA #####

        #Z-score "X" inputs.
        if newData:
            X_train_mean=np.nanmean(X_train,axis=0) #Mean of training data
            X_train_std=np.nanstd(X_train,axis=0) #Stdev of training data
            X_train=(X_train-X_train_mean)/X_train_std #Z-score training data
            X_test=(X_test-X_train_mean)/X_train_std #Preprocess testing data in same manner as training data
            X_valid=(X_valid-X_train_mean)/X_train_std #Preprocess validation data in same manner as training data

            with open(save_folder + dataset + 'latent12FRScore.pickle', 'wb') as f:
                pickle.dump([X_train,X_test,X_valid], f,protocol=4)
        else:
            with open(save_folder + dataset + 'latent12FRScore.pickle','rb') as f:
                X_train,X_test,X_valid = pickle.load(f)


        #Zero-center outputs
        if zeroCenterOutput:
            y_train_mean = np.nanmean(y_train, axis=0)  # Mean of training data outputs
            y_train = y_train - y_train_mean  # Zero-center training output
            y_test = y_test - y_train_mean  # Preprocess testing data in same manner as training data
            y_valid = y_valid - y_train_mean  # Preprocess validation data in same manner as training data



        #Z-score outputs (for SVR)
        if normalizeOutputByStd:
            y_train_std = np.nanstd(y_train, axis=0)
            y_zscore_train = y_train / y_train_std
            y_zscore_test = y_test / y_train_std
            y_zscore_valid = y_valid / y_train_std

        y_mean = np.mean(y_train)
        print('totalMSE: ', np.mean((y_train - y_mean) ** 2))
        y_mean = np.mean(y_valid)
        print('totalMSEvalid: ', np.mean((y_valid - y_mean) ** 2))





        ################# DECODING #################

        #Add actual train/valid/test data to lists (for saving)
        y_test_all.append(y_test)
        y_train_all.append(y_train)
        y_valid_all.append(y_valid)



        ###### WIENER FILTER ######
        if run_wf:
            #Note - the Wiener Filter has no hyperparameters to fit, unlike all other methods

            #Declare model
            model_wf=WienerFilterDecoder()
            #Fit model on training data
            model_wf.fit(X_flat_train,y_train)
            #Get test set predictions
            y_test_predicted_wf=model_wf.predict(X_flat_test)
            #Get R2 of test set (mean of x and y values of position/velocity)
            mean_r2_wf[i]=np.mean(get_R2(y_test,y_test_predicted_wf))
            #Print R2 values on test set
            R2s_wf=get_R2(y_test,y_test_predicted_wf)
            print('R2s_wf:', R2s_wf)
            rho_wf = get_rho(y_test, y_test_predicted_wf)
            print('Rho_wf:', rho_wf)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_wf_all.append(y_test_predicted_wf)
            y_train_pred_wf_all.append(model_wf.predict(X_flat_train))
            y_valid_pred_wf_all.append(model_wf.predict(X_flat_valid))


            ###### WIENER CASCADE ######
        if run_wc:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting (here, degree)
            def wc_evaluate(degree):
                model_wc=WienerCascadeDecoder(degree) #Define model
                model_wc.fit(X_flat_train,y_train) #Fit model
                y_valid_predicted_wc=model_wc.predict(X_flat_valid) #Validation set predictions
                return np.mean(get_R2(y_valid,y_valid_predicted_wc)) #R2 value of validation set (mean over x and y position/velocity)

            #Do bayesian optimization
            wcBO = BayesianOptimization(wc_evaluate, {'degree': (1, 5.01)}, verbose=0) #Define Bayesian optimization, and set limits of hyperparameters
            wcBO.maximize(init_points=3, n_iter=3) #Set number of initial runs and subsequent tests, and do the optimization
            best_params=wcBO.max['params'] #Get the hyperparameters that give rise to the best fit
            degree=best_params['degree']
            #         print("degree=", degree)

            ### Run model w/ above hyperparameters

            model_wc=WienerCascadeDecoder(degree) #Declare model
            model_wc.fit(X_flat_train,y_train) #Fit model on training data
            y_test_predicted_wc=model_wc.predict(X_flat_test) #Get test set predictions
            mean_r2_wc[i]=np.mean(get_R2(y_test,y_test_predicted_wc)) #Get test set R2 (mean across x and y position/velocity)
            #Print R2 values on test set
            R2s_wc=get_R2(y_test,y_test_predicted_wc)
            print('R2s_wc:', R2s_wc)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_wc_all.append(y_test_predicted_wc)
            y_train_pred_wc_all.append(model_wc.predict(X_flat_train))
            y_valid_pred_wc_all.append(model_wc.predict(X_flat_valid))


        ##### Dense (Feedforward) NN ######
        if run_dnn:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            X_train = np.expand_dims(X_train,-1)
            X_test = np.expand_dims(X_test,-1)
            X_valid = np.expand_dims(X_valid,-1)
            def dnn_evaluate(num_units,frac_dropout,LA2,lr):
                num_units=int(num_units)
                LA2 = float(LA2)
                lr = float(lr)
                frac_dropout=float(frac_dropout)
                n_epochs=int(20)
                layers = []
                for i in range(numlayers):
                    layers.append(num_units)
                model_dnn=DenseNNRegression_Hanlin(units=layers,dropout=frac_dropout,num_epochs=n_epochs,LA2=np.power(10,LA2), lr=np.power(10,lr))
                model_dnn.fit(X_train,y_train,X_valid,y_valid)
                y_valid_predicted_dnn=model_dnn.predict(X_valid)
                return np.mean(get_R2(y_valid,y_valid_predicted_dnn))

            #Do bayesian optimization
            dnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (32,128), 'frac_dropout': (0,.7), 'LA2': (-1,-4), 'lr': (np.log(0.2), np.log(0.005))})
            dnnBO.maximize(init_points=40, n_iter=20, kappa=10)
            best_params=dnnBO.max['params']
            frac_dropout=float(best_params['frac_dropout'])
            LA2=np.float(best_params['LA2'])
            num_units=np.int(best_params['num_units'])
            lr = np.float(best_params['lr'])

            # Run model w/ above hyperparameters
            layers = []
            for i in range(numlayers):
                layers.append(num_units)
            model_dnn=DenseNNRegression_Hanlin(units=layers,dropout=frac_dropout,LA2=np.power(10,LA2),num_epochs=20,lr=np.power(10,lr),savePath =RawDatafileName + saveName )
            model_dnn.fit(X_train,y_train,X_valid,y_valid)
            y_test_predicted_dnn=model_dnn.predict(X_test)
            mean_r2_dnn[i]=np.mean(get_R2(y_test,y_test_predicted_dnn))
            #Print R2 values on test set
            R2s_dnn=get_R2(y_test,y_test_predicted_dnn)
            print('R2s:', R2s_dnn)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_dnn_all.append(y_test_predicted_dnn)
            y_train_pred_dnn_all.append(model_dnn.predict(X_train))
            y_valid_pred_dnn_all.append(model_dnn.predict(X_valid))

        if run_dnn_ISA:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
            y_mean = np.mean(y_valid[:,24:36])
            print('totalMSEvalid2436: ', np.mean((y_valid[:,24:36] - y_mean) ** 2))
            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            X_train = np.expand_dims(X_train,-1)
            X_test = np.expand_dims(X_test,-1)
            X_valid = np.expand_dims(X_valid,-1)
            def dnn_evaluate(num_units,frac_dropout,LA2):
                num_units=int(num_units)
                LA2 = float(LA2)
                frac_dropout=float(frac_dropout)
                n_epochs=int(20)
                layers = []
                for i in range(numlayers):
                    layers.append(num_units)
                model_dnn=DenseNNRegression_Hanlin_IntSuperAdd(units=layers,dropout=frac_dropout,num_epochs=n_epochs,LA2=np.power(10,LA2))
                model_dnn.fit(X_train,y_train,X_valid,y_valid)
                y_valid_predicted_dnn=model_dnn.predict(X_valid)
                return np.mean(get_R2(y_valid[:,24:36],y_valid_predicted_dnn[2]))

            #Do bayesian optimization
            dnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (64,512), 'frac_dropout': (0,.5), 'LA2': (-6,-1)})
            dnnBO.maximize(init_points=20, n_iter=20, kappa=10)
            best_params=dnnBO.max['params']
            frac_dropout=float(best_params['frac_dropout'])
            LA2=np.float(best_params['LA2'])
            num_units=np.int(best_params['num_units'])

            # Run model w/ above hyperparameters

            model_dnn=DenseNNRegression_Hanlin_IntSuperAdd(units=[num_units,num_units],dropout=frac_dropout,LA2=np.power(10,LA2),num_epochs=20,savePath =RawDatafileName + saveName )
            model_dnn.fit(X_train,y_train,X_valid,y_valid)
            y_test_predicted_dnn=model_dnn.predict(X_test)
            mean_r2_dnn[i]=np.mean(get_R2(y_test[:,24:36],y_test_predicted_dnn[2]))
            #Print R2 values on test set
            R2s_dnn=get_R2(y_test[:,24:36],y_test_predicted_dnn[2])
            print('R2s:', R2s_dnn)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_dnn_all.append(y_test_predicted_dnn[2])
            y_train_pred_dnn_all.append(model_dnn.predict(X_train)[2])
            y_valid_pred_dnn_all.append(model_dnn.predict(X_valid)[2])



        ##### SIMPLE RNN ######
        if run_rnn:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            def rnn_evaluate(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
                model_rnn.fit(X_train,y_train)
                y_valid_predicted_rnn=model_rnn.predict(X_valid)
                return np.mean(get_R2(y_valid,y_valid_predicted_rnn))

            #Do bayesian optimization
            rnnBO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            rnnBO.maximize(init_points=20, n_iter=20, kappa=10)
            best_params=rnnBO.res['max']['max_params']
            frac_dropout=float(best_params['frac_dropout'])
            n_epochs=np.int(best_params['n_epochs'])
            num_units=np.int(best_params['num_units'])

            # Run model w/ above hyperparameters

            model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
            model_rnn.fit(X_train,y_train)
            y_test_predicted_rnn=model_rnn.predict(X_test)
            mean_r2_rnn[i]=np.mean(get_R2(y_test,y_test_predicted_rnn))
            #Print R2 values on test set
            R2s_rnn=get_R2(y_test,y_test_predicted_rnn)
            print('R2s:', R2s_rnn)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_rnn_all.append(y_test_predicted_rnn)
            y_train_pred_rnn_all.append(model_rnn.predict(X_train))
            y_valid_pred_rnn_all.append(model_rnn.predict(X_valid))

            ##### GRU ######
        if run_gru:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            def gru_evaluate(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_gru=GRUDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
                model_gru.fit(X_train,y_train)
                y_valid_predicted_gru=model_gru.predict(X_valid)
                return np.mean(get_R2(y_valid,y_valid_predicted_gru))

            #Do bayesian optimization
            gruBO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            gruBO.maximize(init_points=20, n_iter=20,kappa=10)
            best_params=gruBO.res['max']['max_params']
            frac_dropout=float(best_params['frac_dropout'])
            n_epochs=np.int(best_params['n_epochs'])
            num_units=np.int(best_params['num_units'])

            # Run model w/ above hyperparameters

            model_gru=GRUDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
            model_gru.fit(X_train,y_train)
            y_test_predicted_gru=model_gru.predict(X_test)
            mean_r2_gru[i]=np.mean(get_R2(y_test,y_test_predicted_gru))
            #Print test set R2 values
            R2s_gru=get_R2(y_test,y_test_predicted_gru)
            print('R2s:', R2s_gru)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_gru_all.append(y_test_predicted_gru)
            y_train_pred_gru_all.append(model_gru.predict(X_train))
            y_valid_pred_gru_all.append(model_gru.predict(X_valid))


            ##### LSTM ######
        if run_lstm:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
            X_train = [X_train[:, :, np.squeeze(layerIndex == i + 1)] for i in range(18)]
            X_valid = [X_valid[:, :, np.squeeze(layerIndex == i + 1)] for i in range(18)]
            X_test= [X_test[:, :, np.squeeze(layerIndex == i + 1)] for i in range(18)]

            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            num_units = 10;
            def lstm_evaluate(frac_dropout,num_units=num_units,n_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act):

                num_units=float(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_lstm=LSTMClassification(units=num_units,dropout=np.power(frac_dropout,1),num_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act)
                model_lstm.fit(X_train,y_train,X_valid,y_valid,NRUPL,NDPL,NAPL)
                y_valid_predicted_lstm=model_lstm.predict(X_valid)
                return np.mean(y_valid==y_valid_predicted_lstm)

            #Do bayesian optimization
            lstmBO = BayesianOptimization(lstm_evaluate, {'frac_dropout': (0.05,0.5),'n_epochs':(10,30)})
            lstmBO.maximize(init_points=3, n_iter=3, kappa=10)
            best_params=lstmBO.max['params']
            frac_dropout=np.power(float(best_params['frac_dropout']),1)
            n_epochs=np.float(best_params['n_epochs'])
            # Run model w/ above hyperparameters

            model_lstm=LSTMClassification(units=num_units,dropout=frac_dropout,num_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act)
            model_lstm.fit(X_train,y_train,X_valid,y_valid,NRUPL,NDPL,NAPL)
            y_test_predicted_lstm=model_lstm.predict(X_test)
            mean_r2_lstm[i]=np.mean((y_test==y_test_predicted_lstm))
            #Print test set R2
            R2s_lstm=np.mean(y_test==y_test_predicted_lstm)
            print('R2s:', R2s_lstm)
            rho_lstm = np.mean(y_test==y_test_predicted_lstm)
            print('Rho_lstm:', rho_lstm)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_lstm_all.append(y_test_predicted_lstm)
            y_train_pred_lstm_all.append(model_lstm.predict(X_train))
            y_valid_pred_lstm_all.append(model_lstm.predict(X_valid))

        print ("\n") #Line break after each fold
        time_elapsed=time.time()-t1 #How much time has passed

        if run_lstm_sim:

            ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
            #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
            #as a function of the hyperparameter we are fitting
            num_units = 100
            n_epochs  = 10

            def lstm_evaluate(frac_dropout,LA2,num_units=num_units,n_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act):

                num_units=int(num_units)
                LA2 = float(LA2)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_lstm=LSTMClassification_sim(units=num_units,LA2=np.power(10,LA2), dropout=np.power(frac_dropout,1),num_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act)
                model_lstm.fit(X_train,y_train,X_valid,y_valid,NRUPL,NDPL,NAPL)
                y_valid_predicted_lstm=model_lstm.predict(X_valid)
                return np.mean(y_valid==y_valid_predicted_lstm)

            #Do bayesian optimization
            lstmBO = BayesianOptimization(lstm_evaluate, {'frac_dropout': (0.05,0.2),'LA2':(-8.,-2)})
            lstmBO.maximize(init_points=8, n_iter=2, kappa=10)
            best_params=lstmBO.max['params']
            frac_dropout=np.power(float(best_params['frac_dropout']),1)
            LA2=np.float(best_params['LA2'])
            # Run model w/ above hyperparameters

            model_lstm=LSTMClassification_sim(units=num_units,LA2=LA2,dropout=frac_dropout,num_epochs=n_epochs,batch_size=batch_size,lr=lr,final_act=final_act)
            model_lstm.fit(X_train,y_train,X_valid,y_valid,NRUPL,NDPL,NAPL)
            y_test_predicted_lstm=model_lstm.predict(X_test)
            y_test = np.argmax(y_test,axis=1)
            mean_r2_lstm[i]=np.mean((y_test==y_test_predicted_lstm))
            #Print test set R2
            R2s_lstm=np.mean(y_test==y_test_predicted_lstm)
            print('R2s:', R2s_lstm)
            rho_lstm = np.mean(y_test==y_test_predicted_lstm)
            print('Rho_lstm:', rho_lstm)
            #Add predictions of training/validation/testing to lists (for saving)
            y_pred_lstm_all.append(y_test_predicted_lstm)
            y_train_pred_lstm_all.append(model_lstm.predict(X_train))
            y_valid_pred_lstm_all.append(model_lstm.predict(X_valid))

        print ("\n") #Line break after each fold
        time_elapsed=time.time()-t1 #How much time has passed



        ###### SAVE RESULTS #####
        #Note that I save them after every cross-validation fold rather than at the end in case the code/computer crashes for some reason while running

        #Only save results for the decoder we chose to run
        if run_wf:
            with open(save_folder+dataset+'_results_wf2.pickle','wb') as f:
                pickle.dump([mean_r2_wf,y_pred_wf_all,y_train_pred_wf_all,y_valid_pred_wf_all],f)
            io.savemat(save_folder + dataset + '_results_wf2.mat', mdict={'mean_r2_wf': mean_r2_wf, 'y_pred_wf_all':y_pred_wf_all, 'y_train_pred_wf_all':y_train_pred_wf_all, 'y_valid_pred_wf_all':y_valid_pred_wf_all })
            #scipy.io.savemat(save_folder+dataset+'_results_wf2.mat', mdict={'':,'':,'':,'':,})
        if run_wc:
            with open(save_folder+dataset+'_results_wc2.pickle','wb') as f:
                pickle.dump([mean_r2_wc,y_pred_wc_all,y_train_pred_wc_all,y_valid_pred_wc_all],f)

        if run_xgb:
            with open(save_folder+dataset+'_results_xgb2.pickle','wb') as f:
                pickle.dump([mean_r2_xgb,y_pred_xgb_all,y_train_pred_xgb_all,y_valid_pred_xgb_all,time_elapsed],f)

        if run_dnn or run_dnn_ISA:
            with open(save_folder+dataset+'_results_dnn2.pickle','wb') as f:
                pickle.dump([mean_r2_dnn,y_pred_dnn_all,y_train_pred_dnn_all,y_valid_pred_dnn_all,time_elapsed],f)

            io.savemat(save_folder + dataset + RawDatafileName + saveName + str(numlayers) + 'layers' + '_results_tfdnn.mat',
                       mdict={'mean_r2_tfdnn': mean_r2_dnn, 'y_pred_dnn_all': y_pred_dnn_all,
                              'y_train_pred_tfdnn_all': y_train_pred_dnn_all,
                              'y_valid_pred_tfdnn_all': y_valid_pred_dnn_all, 'time_elapsed': time_elapsed,
                              'best_params': best_params})

        if run_rnn:
            with open(save_folder+dataset+'_results_rnn2.pickle','wb') as f:
                pickle.dump([mean_r2_rnn,y_pred_rnn_all,y_train_pred_rnn_all,y_valid_pred_rnn_all,time_elapsed],f)

        if run_gru:
            with open(save_folder+dataset+'_results_gru2.pickle','wb') as f:
                pickle.dump([mean_r2_gru,y_pred_gru_all,y_train_pred_gru_all,y_valid_pred_gru_all,time_elapsed],f)

        if run_lstm or run_lstm_sim:
            with open(save_folder+dataset+'_results_lstm2.pickle','wb') as f:
                pickle.dump([mean_r2_lstm,y_pred_lstm_all,y_train_pred_lstm_all,y_valid_pred_lstm_all,time_elapsed],f)
            io.savemat(save_folder + dataset + RawDatafileName + str(KinV) + '_part_' + str(Neural_ensemble_part) + '_results_lstmBatch.mat', mdict={'mean_r2_lstm':mean_r2_lstm, 'y_pred_lstm_all':y_pred_lstm_all, 'y_train_pred_lstm_all':y_train_pred_lstm_all, 'y_valid_pred_lstm_all':y_valid_pred_lstm_all,'time_elapsed':time_elapsed,'best_params':best_params })

        if run_svr:
            with open(save_folder+dataset+'_results_svr2.pickle','wb') as f:
                pickle.dump([mean_r2_svr,y_pred_svr_all,y_train_pred_svr_all,y_valid_pred_svr_all,time_elapsed],f)


    #Save ground truth results
    with open(save_folder+dataset+'_ground_truth.pickle','wb') as f:
        pickle.dump([y_test_all,y_train_all,y_valid_all],f)
    io.savemat(save_folder + dataset + RawDatafileName + saveName + str(numlayers) +'layers' + '_ground_truth.mat', mdict={'y_test_all':y_test_all,'y_train_all':y_train_all,'y_valid_all':y_valid_all})
    print("time_elapsed:",time_elapsed)




