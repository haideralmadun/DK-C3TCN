import pandas as pd
from numpy import array
from DK_C3TCN_model import DK_C3TCN_model
from summarize_average_performance import summarize_average_performance



# load the dataset
#dataset = pd.read_csv(r'changhua_Water_Rainfall.csv')

dataset = pd.read_csv(r'tunxi 1981-2016_interpolated.csv')



# remove date colume 
dataset_new= dataset.iloc[:,1:]

dataset_new= dataset_new.dropna()



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_new_no = scaler.fit_transform(dataset_new)


# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(dataset_new['streamflow'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)




# convert data to dataframe to enable split it for three part

df = pd.DataFrame(dataset_new_no)


# split data to x.train y.train and  x.test y.test

dataset_train = df.iloc[:37622,:].values
dataset_val   = df.iloc[37622:42997,:].values
dataset_test  = df.iloc[42997:,:].values




# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)






# choose a number of time steps
n_steps_in, n_steps_out = 12, 6



# convert into input/output
X_train, y_train = split_sequences(dataset_train, n_steps_in, n_steps_out)
print(X_train.shape, y_train.shape)
X_val, y_val = split_sequences(dataset_val , n_steps_in, n_steps_out)
print(X_val.shape, y_val.shape)
X_test, y_test = split_sequences(dataset_test, n_steps_in, n_steps_out)
print(X_test.shape, y_test.shape)


n_features = X_train.shape[2]






# Initialize the DK-TD3S2T model
model = DK_C3TCN_model( n_steps_in=n_steps_in, n_steps_out=n_steps_out, n_features=n_features)

# Show model summary
model.summary()


from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
										mode ="min", patience = 15,
										restore_best_weights = True)




# Fit model
history = model.fit(
    X_train,
    y_train,
    batch_size=200,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[earlystopping],
    verbose=1
)





    
# Predict on test set
y_pred = model.predict(X_test, batch_size=200, verbose=1)

# Inverse scale
y_pred_unscaled = scaler_pred.inverse_transform(y_pred)
y_test_unscaled = scaler_pred.inverse_transform(y_test)

# Summarize performance
summarize_average_performance('DK-C3TCN', y_test_unscaled, y_pred_unscaled)






##### Load the model

with open('model.json','r') as f:
    json = f.read()
model.load_weights("DK-C3STCN_tunxi_learning_rate.h5")

# summarize model.
model.summary()


