import pandas as pd
import numpy as np
import pickle

from matplotlib import pyplot as plt

#data source https://raw.githubusercontent.com/organisciak/names/master/data/us-names-by-year.csv

trainTestSplit=0.6
useNewData=False
# filter out chosen names here (don't ever commit code with this!!)
#can add any names we want to predict on here
#may do this differntly - using input from below
holdout_names=list()

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
            
            for year in range(1991,2013):
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
    
    pickle.dump( X, open( "X.p", "wb" ) )
    pickle.dump( Y, open( "Y.p", "wb" ) )
    pickle.dump( name_list, open( "name_list.p", "wb" ) )
    
    

else:
    X=pickle.load( open( "X.p", "rb" ) )
    Y=pickle.load( open( "Y.p", "rb" ) )
    name_list = pickle.load( open( "name_list.p", "rb" ) )
    
#TODO - filter out holdout names here
    
X_train = X[0:int(trainTestSplit * np.shape(X)[0]),]
X_test = X[int(trainTestSplit * np.shape(X)[0]) + 1: np.shape(X)[0],]

Y_train = Y[0:int(trainTestSplit * np.shape(Y)[0]),]
Y_test = Y[int(trainTestSplit * np.shape(Y)[0]) + 1: np.shape(Y)[0],]


#%%
#plot a random instance
rand = np.random.randint(len(X))
train_inst = np.append(X[rand],Y[rand])


plt.plot(list(range(1910,1910+len(train_inst))),train_inst)
plt.title("occurance of name "+name_list[rand])
plt.tight_layout()



#plot a given name
givenName=input("please enter name to look up...")
rand = name_list.index(givenName)
train_inst = np.append(X[rand],Y[rand])


plt.plot(list(range(1910,1910+len(train_inst))),train_inst)
plt.title("occurance of name "+name_list[rand])
plt.tight_layout()
#