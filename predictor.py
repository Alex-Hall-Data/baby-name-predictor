import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import math

from matplotlib import pyplot as plt

tf.reset_default_graph()
#data source https://raw.githubusercontent.com/organisciak/names/master/data/us-names-by-year.csv

trainTestSplit=0.6
lagYears=22 #number of years to predict over
useNewData=False
scale = True
train_model= True # training up a model or just predicting


if useNewData:
    raw_data = pd.read_csv("us-names-by-year.csv")
    raw_data=raw_data.sample(frac=1) # shuffle the df
    
    male_names = raw_data[raw_data['sex']=="M"]
    
    female_names = raw_data[raw_data['sex']=="F"]
    
    #for testing only
    #male_names = male_names[male_names['name'].isin(["Matthew","Mark","Luke"])]
    #female_names = female_names[ female_names['name'].isin(["Hannah","Sarah"])]
    
    
    
    #%%
    #reshape the data - each name gets a list of popularities for X and y_true
    def data_reshape(dataframe):
        reshaped_x=list()
        reshaped_y=list()
        name_list=list()
        counter =0
    
        for name in pd.unique(dataframe['name']):
            name_list.append(name)
            name_df = dataframe[raw_data['name']==name]
            name_df=name_df.sort_values(by=['year'])
            
            print(counter)
            counter=counter+1
            
            #populate arrays for each name - years with zero occurances have zero value (in trycatch)
            name_x=list()
            name_y=list()
            for year in range(1910,1990):
                try:
                    name_x.append( name_df[name_df['year'] == year]['count'].item())
                except:
                    name_x.append(0)
            
            for year in range(1933,2013): #note - shifted rather than just taking last few years -maintains x and y as same shape
                try:
                    name_y.append( name_df[name_df['year'] == year]['count'].item())
                except:
                    name_y.append(0)
    
            
            reshaped_x.append(name_x)
            reshaped_y.append(name_y)
    
            
        reshaped_x=np.asarray(reshaped_x)
        reshaped_y=np.asarray(reshaped_y)
        
        return reshaped_x , reshaped_y,name_list
            
    #
    male_names_x , male_names_y ,male_name_list = data_reshape(male_names)
    female_names_x , female_names_y , female_name_list  = data_reshape(female_names)
    
    #can use these three objects to see popularity of specific names (names list is in same order as X and Y)
    X = np.append(male_names_x,female_names_x,axis=0)
    Y = np.append(male_names_y,female_names_y,axis=0)
    name_list=male_name_list+female_name_list 
    
    
        #get rid of Nans    
    X[np.isnan(X)]=0
    Y[np.isnan(Y)]=0
    
    def scale_data(data_x , data_y):
        for i in range(np.shape(data_x)[0]):
            max_value = max(np.append(data_x[i,] , data_y[i,]))
            min_value = min(np.append(data_x[i,] , data_y[i,]))
            
            data_x[i,] = (data_x[i,] - min_value) / (max_value-min_value)
            data_y[i,] = (data_y[i,] - min_value) / (max_value-min_value)
            
        return data_x , data_y
    
    #scale the data
    if(scale):
        X , Y = scale_data(X,Y)
        
    #write to disk    
    pickle.dump( X, open( "X.p", "wb" ) )
    pickle.dump( Y, open( "Y.p", "wb" ) )
    pickle.dump( name_list, open( "name_list.p", "wb" ) )

#use previously saved data
else:
    X=pickle.load( open( "X.p", "rb" ) )
    Y=pickle.load( open( "Y.p", "rb" ) )
    name_list = pickle.load( open( "name_list.p", "rb" ) )
 
#get rid of Nans    
X[np.isnan(X)]=0
Y[np.isnan(Y)]=0




#%%

#select the name to lookup and plot its known data
#keep the given name data to use for prediction later
givenName=input("please enter name to look up...")
rand = name_list.index(givenName)
givenNameData = np.append(X[rand][0:lagYears],Y[rand])


plt.plot(list(range(1910,1910+len(givenNameData))),givenNameData)
plt.title("occurance of name "+name_list[rand])
plt.tight_layout()
#
#%%
#drop the lookup name from the dataset and build train and test set
X=np.delete(X,rand,0)
Y=np.delete(Y,rand,0)
X_train = X[0:int(trainTestSplit * np.shape(X)[0]),]
X_test = X[int(trainTestSplit * np.shape(X)[0]) + 1: np.shape(X)[0],]

Y_train = Y[0:int(trainTestSplit * np.shape(Y)[0]),]
Y_test = Y[int(trainTestSplit * np.shape(Y)[0]) + 1: np.shape(Y)[0],]

#%%
#build up rnn

# Just one feature, the time series
num_inputs = 1
# Just one output, predicted time series
num_outputs = 1


# Size of the batch of data
batch_size = 1
# how many iterations to go through (training steps), you can play with this
num_train_iterations = math.floor(len(X_train)/batch_size)

num_time_steps = np.shape(X_train)[1]
num_y_time_steps = np.shape(Y_test)[1]


X_in = tf.placeholder(tf.float32, [None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32, [None,num_y_time_steps,num_outputs])

#learning rate 0.00005 is best so far
k = 0.00005

#define RNN cell
n_neurons = 50
n_layers = 10
cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)

#dynamic rnn cell
outputs, states = tf.nn.dynamic_rnn(cell, X_in, dtype=tf.float32)

#define loss function
global_step = tf.Variable(0, trainable=False,dtype=tf.int64)
loss = tf.reduce_mean(tf.square(outputs - y)) # RMSE - output of last cell

#define optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=k)
learning_step = optimizer.minimize(loss)

#%%
#run the rnn
init = tf.global_variables_initializer()


saver = tf.train.Saver()

# In[26]:


if (train_model):  
    with tf.Session() as sess:
          sess.run(init)
          
          loss_list=list()
          iteration_list=list()
          
          for iteration in range(num_train_iterations):
                      
              X_batch = np.reshape(X_train[iteration],(batch_size,num_time_steps,num_inputs))
              y_batch = np.reshape(Y_train[iteration],(batch_size,num_y_time_steps,num_outputs))
              sess.run(learning_step, feed_dict={X_in: X_batch, y: y_batch})
              
              
              
              if iteration % 10 == 0:
                  
                  
                  #rmse = loss.eval(feed_dict={X_in: X_batch, y: y_batch})
                     # print(iteration, "\tRMSE:", rmse)
                  
                  #accuracy on test set
                  test_mse=loss.eval(feed_dict={X_in: np.reshape(X_test,( np.shape(X_test)[0],num_time_steps,1)),y:np.reshape(Y_test,( np.shape(Y_test)[0],num_time_steps,1))})
                     # print(iteration, "\tRMSE on test:", test_rmse)
                  
                  loss_list.append(test_mse)
                  iteration_list.append(iteration)
                  
                  print("iteration " + str(iteration) + " test mse " + str(test_mse))
      
          # Save Model for Later
          saver.save(sess, "./rnn_time_series_model")

 

#%%
#predict on the target name
    
with tf.Session() as sess:                          
    saver.restore(sess, "./rnn_time_series_model")   
    
   
    X_new = np.reshape(givenNameData[0:80],(1,num_time_steps,num_inputs))
    y_true = np.reshape(givenNameData[lagYears:],(1,num_time_steps,num_inputs))
    y_pred = sess.run(outputs, feed_dict={X_in: X_new})
    
    #y_pred_list=sess.run(outputs,feed_dict={X_in:np.reshape(x_test,(len(x_test),np.shape(x_test)[1],1))})
prediction = y_pred[0,:,0]
# In[28]:
#plot example prediction
plt.figure(0)
plt.title("Testing Example")

# Test Instance
plt.plot(list(range(0,num_time_steps)),X_new[0],label="Input")
plt.plot(list(range(lagYears,num_time_steps+lagYears)), y_true[0],'ro',label="Actual")

# Target to Predict
plt.plot(list(range(lagYears,lagYears+num_time_steps)), prediction,'bs', label="Predicted")


plt.xlabel("Time")
plt.legend()
#plt.tight_layout()

axes = plt.gca()
axes.set_ylim(0,1.1)