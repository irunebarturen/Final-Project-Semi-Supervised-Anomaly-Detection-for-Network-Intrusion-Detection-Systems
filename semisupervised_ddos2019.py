import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Input, AlphaDropout, Lambda, RepeatVector
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,  f1_score, make_scorer, precision_score, recall_score
from sklearn.svm import OneClassSVM
from model_evaluation import *

##OPTION 1: USING 31 FEATURES
#31 features selected from the preprocessing file
features = [' Average Packet Size',
 ' Avg Bwd Segment Size',
 ' Avg Fwd Segment Size',
 ' Bwd Header Length',
 ' Bwd IAT Mean',
 ' Bwd Packet Length Mean',
 ' Bwd Packet Length Min',
 ' Destination Port',
 ' Flow Packets/s',
 ' Fwd Header Length',
 ' Fwd Header Length.1',
 ' Fwd Packet Length Max',
 ' Fwd Packet Length Mean',
 ' Fwd Packet Length Min',
 ' Inbound',
 ' Max Packet Length',
 ' Min Packet Length',
 ' Packet Length Mean',
 ' Packet Length Std',
 ' Packet Length Variance',
 ' Protocol',
 ' Source Port',
 ' Subflow Bwd Bytes',
 ' Subflow Fwd Bytes',
 ' Total Backward Packets',
 ' act_data_pkt_fwd',
 ' min_seg_size_forward',
 'Bwd Packet Length Max',
 'Fwd Packets/s',
 'Init_Win_bytes_forward',
 'Total Length of Fwd Packets']

print(f"Number of columns: {len(features)}")

data_benign_features = pd.read_csv('benign_01_12.csv', usecols=features)

print(f"Features type: {data_benign_features.dtypes}")

##OPTION 2: USING ALL THE FEATURES
#data_benign_features = pd.read_csv('benign_01_12.csv')
#print(f"Number of columns: {len(data_benign_features.columns)}")

#print(f"Features type: {data_benign_features.dtypes}")

#Drop unnecessary features
#data_benign_features = data_benign_features.drop(['Unnamed: 0',' Label', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP', 'File', ' Label']], axis=1)
#features=list(data_benign_features.columns)

#####

## No need to encode any feature as they are all numerical

##Drop rows with nan and inf values

print(np.isnan(data_benign_features).sum(), "NaNs in X_train")
print(np.isinf(data_benign_features).sum(), "Infs in X_train")

data_benign_features.replace([np.inf, -np.inf], np.nan, inplace=True)
data_benign_features.dropna(inplace=True)

## Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_benign_features)

X_train = pd.DataFrame(X_scaled, columns=data_benign_features.columns, index=data_benign_features.index)



####ONE-CLASS SVM

# Custom F1 scoring function for anomaly detection
# Converts One-Class SVM output: -1 (anomaly), 1 (normal) → 1 (anomaly), 0 (normal)
def f1_scorer(y_true, y_pred):
    y_pred = (y_pred == -1).astype(int)  
    return f1_score(y_true, y_pred, zero_division=0)

# Wrap the custom scorer to use it in GridSearchCV
scorer = make_scorer(f1_scorer, greater_is_better=True)

# Define hyperparameter grid for One-Class SVM
param_grid = {
    'nu': [0.01, 0.05, 0.1, 0.2, 0.5],
    'gamma':  ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Initialize One-Class SVM model
ocsvm = OneClassSVM()

# Perform grid search with cross-validation using the custom F1 scorer
grid = GridSearchCV(
    estimator=ocsvm,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,  # cross-validation folds
    verbose=1,
    n_jobs=-1
)

# Fit the model using training data (labels are ignored during training, used only for scoring)
y_train = np.zeros(X_train.shape[0])
grid.fit(X_train, y_train) 

print("Best parameters found:", grid.best_params_)

# Use best model for prediction
best_model = grid.best_estimator_


# Evaluate the model on test data
def test_file_ml(features, scaler, best_model, filename, seed=42):
    
    path =  os.path.join(r"datasets/CICDDoD2019_test_new/", filename + '.csv') 
 
    sample_frac = 0.1  # Use 10% of each file for testing
    chunksize = 100_000
    sampled_chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=features+[' Label']):
        sampled_chunk = chunk.sample(frac=sample_frac, random_state = seed)
        sampled_chunks.append(sampled_chunk)

    data = pd.concat(sampled_chunks, ignore_index=True)
        
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    X = data.drop(columns=[' Label'])
    y_true = (data[' Label'] != 'BENIGN').astype(int)  # Binary classification: 1 if attack, 0 if benign
    
    X_scaled = scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    y_pred = (y_pred == -1).astype(int)

    # Calculate metrics
    metrics = {
    'Accuracy': accuracy_score(y_true, y_pred),
    'Macro F1': f1_score(y_true, y_pred, average='macro'),
    'Macro precision': precision_score(y_true, y_pred, average='macro'),
    'Macro recall': recall_score(y_true, y_pred, average='macro'),

    'Dataset': filename
}

    return metrics


# Evaluate the model on multiple test files
results = []
file_list = ['Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag', 'Syn']

for filename in file_list:
    print(f"Processing {filename}")
    metrics = test_file_ml(features, scaler, best_model, filename)
    results.append(metrics)


# Convert list of dicts into a DataFrame
metrics_df = pd.DataFrame(results)
# Melt the dataframe to long format for seaborn
melted_df = metrics_df.melt(id_vars='Dataset', var_name='metrics', value_name='value')

#Plot the metrics for each of the files
plt.figure(figsize=(18, 7))
ax = sns.barplot(data=melted_df, x='Dataset', y='value', hue='metrics') 
# Add metric value labels inside each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='center',rotation=90, fontsize=14, color='white'  )
plt.xticks(rotation=45, ha='right')
plt.title('Classification Metrics per Dataset')
ax.legend(loc='best', bbox_to_anchor=(1.02, 0.5), title='Metrics')
plt.tight_layout()
plt.show()

print(metrics_df)

#Compute the average metrics
averages = metrics_df.drop(columns=['Dataset']).mean()
print(averages)

#Save these results 
average_row = averages.to_frame().T  
average_row['Dataset'] = 'Overall Metrics'
metrics_df = pd.concat([metrics_df, average_row], ignore_index=True)
metrics_df.to_csv('results/ocsvm_31features.csv', index=False)

#### DEEP LEARNING MODELS

##AUTOENCODER 0: Simple model with 1 hidden layer (latent space = 4)
# Uses SELU activation and LeCun Normal initializer for self-normalizing networks
def build_autoencoder0(n_features=31, latent_dim=4):
    model = Sequential([
        Input(shape=(n_features,)),

        Dense(latent_dim, activation='selu', kernel_initializer='lecun_normal'),
        Dense(n_features, activation='linear')  # Linear output for regression-style reconstruction
    ])
    return model

##AUTOENCODER 1: Deeper model with symmetric architecture and latent space = 8
# Uses SELU activation and LeCun Normal initializer for self-normalizing networks

def build_autoencoder1(n_features=31, latent_dim=8):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation='selu', kernel_initializer='lecun_normal'),
        Dense(32, activation='selu', kernel_initializer='lecun_normal'),
        Dense(latent_dim, activation='selu', kernel_initializer='lecun_normal'),
        Dense(32, activation='selu', kernel_initializer='lecun_normal'),
        Dense(64, activation='selu', kernel_initializer='lecun_normal'),
        Dense(n_features, activation='linear')  # No activation in output layer
    ])
    return model

n_features=X_train.shape[1]

#Choose between Autoencoder 0 and Autoencoder 1
autoencoder = build_autoencoder0(n_features=n_features, latent_dim =4)
#autoencoder = build_autoencoder1(n_features=n_features, latent_dim=8)


# Compile the model with Mean Absolute Error (MAE) loss and Adam optimizer
optimizer = Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='mae')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True )


# Train the autoencoder with 30% of data used for validation
history=autoencoder.fit(X_train, X_train, epochs = 200, batch_size=32, validation_split= 0.3 ,callbacks=[early_stop])

# Save the trained model
autoencoder.save('feature_selection_31/autoencoder0.keras')
# Reload the model (for future use or evaluation)
autoencoder = load_model('feature_selection_31/autoencoder0.keras')

# Plot training and validation loss curves
def plot_learning_curves(history):
    metrics = [ 'loss']  # Add more metrics if needed
    plt.figure(figsize=(16, 10))
    
    for metric in metrics:
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
    
    plt.title('Learning Curves over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
plot_learning_curves(history)

# Calculate reconstruction loss for training data
loss=losses.mae
reconstructed = autoencoder(X_train)
train_loss = loss(reconstructed, X_train)

# Compute mean and standard deviation of reconstruction loss
train_loss_mean = np.mean(train_loss)
train_loss_std = np.std(train_loss)
print(f"Mean  of Reconstruction Loss: {train_loss_mean:.4f}")
print(f"Standard Deviation of Reconstruction Loss: {train_loss_std:.4f}")

# Set anomaly threshold at 95th percentile of training loss
threshold = np.percentile(train_loss, 95)
print(f"Anomaly threshold (95th percentile): {threshold:.4f}")


# Plot histogram of training reconstruction losses
plt.figure(figsize=(10,6))
plt.hist(train_loss, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(train_loss_mean, color='green', linestyle='--', linewidth=2, label=f'Mean = {train_loss_mean:.4f}')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Percentile 95 = {threshold:.4f}')
plt.title('Histogram of Training Reconstruction Loss')
plt.xlabel('Reconstruction Loss')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


def plot_distribution_separated(y_pred,y_test, optimal_threshold, filename= ''):

    # Separate the predicted probabilities by true class
    probs_class0 = y_pred[y_test == 0]
    probs_class1 = y_pred[y_test == 1]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Class 0
    axes[0].hist(probs_class0, bins=100, color='blue')
    # Plot the decision threshold

    axes[0].axvline(optimal_threshold, color='green', linestyle='--', label=f'Threshold = {optimal_threshold:.2f}')
    axes[0].set_title('Reconstruction error for Benign data')
    axes[0].set_xlabel('Reconstruction error')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Class 1
    axes[1].hist(probs_class1, bins=100, color='coral')
    axes[1].axvline(optimal_threshold, color='green', linestyle='--', label=f'Threshold = {optimal_threshold:.2f}')
    axes[1].set_title('Reconstruction error for Attack data')
    axes[1].set_xlabel('Reconstruction error')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def test_file(features, scaler, autoencoder, loss, threshold, filename, seed=42):
    

    path =  os.path.join(r"datasets/CICDDoD2019_test_new/", filename + '.csv') 
 
    sample_frac = 0.1 #0.15 #0.2 random sample #Try with different proportion depending on the file length
    chunksize = 100_000
    sampled_chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=features + [' Label']):
        sampled_chunk = chunk.sample(frac=sample_frac, random_state = seed)
        sampled_chunks.append(sampled_chunk)

    data = pd.concat(sampled_chunks, ignore_index=True)
        
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    X = data.drop(columns=[' Label'])
    y_true = (data[' Label'] != 'BENIGN').astype(int)  # Binary classification: 1 if attack, 0 if benign
    
    X_scaled = scaler.transform(X)
    y_pred = autoencoder(X_scaled)
    y_scores = loss(y_pred, X_scaled).numpy()
    y_classes = (y_scores > threshold).astype(int)
    # Calculate metrics
    metrics = {
    'Accuracy': accuracy_score(y_true, y_classes),
    'Macro F1': f1_score(y_true, y_classes, average='macro'),
    'Macro precision': precision_score(y_true, y_classes, average='macro'),
    'Macro recall': recall_score(y_true, y_classes, average='macro'),

    'Dataset': filename
}

    plot_distribution_separated(y_scores, y_true, threshold, filename)

    return metrics


# Evaluate the model on multiple test files
results = []
file_list = ['Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag', 'Syn']

for filename in file_list:
    print(f"Processing {filename}")
    metrics = test_file(features, scaler, autoencoder, loss, threshold, filename)
    results.append(metrics)

# Convert list of dicts into a DataFrame
metrics_df = pd.DataFrame(results)
# Melt the dataframe to long format for seaborn
melted_df = metrics_df.melt(id_vars='Dataset', var_name='metrics', value_name='value')

#Plot the metrics for each of the files
plt.figure(figsize=(18, 7))
ax = sns.barplot(data=melted_df, x='Dataset', y='value', hue='metrics') 
# Add metric value labels inside each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='center',rotation=90, fontsize=14, color='white'  )
plt.xticks(rotation=45, ha='right')
plt.title('Classification Metrics per Dataset')
ax.legend(loc='best', bbox_to_anchor=(1.02, 0.5), title='Metrics')
plt.tight_layout()
plt.show()

print(metrics_df)

#Compute the average metrics
averages = metrics_df.drop(columns=['Dataset']).mean()
print(averages)

#Save these results 
average_row = averages.to_frame().T  
average_row['Dataset'] = 'Overall Metrics'
metrics_df = pd.concat([metrics_df, average_row], ignore_index=True)
metrics_df.to_csv('results/autoencoder0_allfeatures.csv', index=False)

####DEEP LEARNING MODEL WITH ORDER

##AUTOENCODER 2: Sequence model using LSTM layers

#Create sliding windows within each split
def create_sequences(X, window_size = 10):
    X_seqs = []
    for i in range(len(X) - window_size + 1):
        X_seqs.append(X[i:i+window_size])
        #y_seqs.append(y[i:i+window_size])
    #return np.array(X_seqs), np.array(y_seqs)
    return np.array(X_seqs)

#Choose seq_length or window size
seq_length = 20
X_train_seq = create_sequences(X_train, seq_length)

# Shuffle the sequences within each split
perm_train = np.random.RandomState(seed=42).permutation(len(X_train_seq))
X_train_seq_s = X_train_seq[perm_train]

print(X_train.shape) 
print(X_train_seq.shape) 

def build_autoencoder2(seq_length, n_features):
    input_layer = Input(shape=(seq_length, n_features))

    # Encoder
    x = LSTM(32, return_sequences=True)(input_layer)
    x = LSTM(16, return_sequences=True)(x)  # Latent sequence representation

    # Decoder (symmetric structure)
    x = LSTM(16, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)

    # Output layer (reconstruct input sequence)
    decoded = TimeDistributed(Dense(n_features))(x)

    model = Model(input_layer, decoded)
    return model

# Compile the model

steps = X_train_seq.shape[1]
n_features = X_train_seq.shape[2]
latent_dim = 32  

# Build the LSTM autoencoder
autoencoder = build_autoencoder2(seq_length=steps, n_features=n_features)

# Compile the model with Mean Absolute Error (MAE) loss and Adam optimizer
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mae')
autoencoder.summary()

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True )

# Train the autoencoder with 30% of data used for validation
history=autoencoder.fit(X_train_seq_s, X_train_seq_s, epochs = 200, batch_size=64, validation_split= 0.3 ,callbacks=[early_stop])

# Save the trained model
autoencoder.save('feature_selection_31/autoencoder2.keras')
# Reload the model (for future use or evaluation)
autoencoder = load_model('feature_selection_31/autoencoder2.keras')

# Plot training and validation loss curves
plot_learning_curves(history)


def recover_array(x_pred_seq,perm_test,l):
    window_size = x_pred_seq.shape[1]
    n_features = x_pred_seq.shape[2]
    x_sum = np.zeros((l,n_features))
    x_count = np.zeros((l,n_features))
    perm_test_inverted = np.argsort(perm_test) #Invert the permutation
    for i,s in list(enumerate(perm_test_inverted)):
        seq = x_pred_seq[s]
        x_sum[i:i + window_size] += seq
        x_count[i:i + window_size,:] += 1
    return x_sum /x_count

X_pred_seq = autoencoder.predict(X_train_seq_s)


l=len(X_train)
X_reconstructed = recover_array(X_pred_seq,perm_train, l)
print(X_reconstructed.shape)

# Calculate reconstruction loss for training data
loss=losses.mae
train_loss = loss(X_reconstructed, X_train)

# Compute mean and standard deviation of reconstruction loss
train_loss_mean = np.mean(train_loss)
train_loss_std = np.std(train_loss)
print(f"Mean  of Reconstruction Loss: {train_loss_mean:.4f}")
print(f"Standard Deviation of Reconstruction Loss: {train_loss_std:.4f}")

# Set anomaly threshold at 95th percentile of training loss
threshold = np.percentile(train_loss, 95)
print(f"Anomaly threshold (95th percentile): {threshold:.4f}")


# Plot histogram of training reconstruction losses
plt.figure(figsize=(10,6))
plt.hist(train_loss, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(train_loss_mean, color='green', linestyle='--', linewidth=2, label=f'Mean = {train_loss_mean:.4f}')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Mean + Std (t) = {threshold:.4f}')
plt.title('Histogram of Training Reconstruction Loss')
plt.xlabel('Reconstruction Loss')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()



def test_file_order(features, scaler, autoencoder, loss, threshold, filename, seed=42):
    

    path =  os.path.join(r"/datasets/CICDDoD2019_test_new/", filename + '.csv') 
    chunksize = 10_000
    n_chunks = 0
    accum_metrics = defaultdict(list)
    
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=features + [' Timestamp',' Label'])):
        if i%2!=0:
            continue
        n_chunks += 1
        data = chunk.sort_values([' Timestamp'])

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        X = data.drop(columns=[' Timestamp',' Label'])
        y_true = (data[' Label'] != 'BENIGN').astype(int)  # Binary classification: 1 if attack, 0 if benign

        X_scaled = scaler.transform(X)
        seq_length = 20
        X_seq = create_sequences(X_scaled, seq_length)

        perm = np.random.RandomState(seed=seed).permutation(len(X_seq))
        X_seq_s = X_seq[perm]
        X_pred_seq = autoencoder(X_seq_s)

        l = len(X_scaled)
        X_reconstructed = recover_array(X_pred_seq, perm, l)
        y_scores = loss(X_reconstructed, X_scaled).numpy()
        y_pred = (y_scores > threshold).astype(int)

        # Compute and store metrics
        accum_metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        accum_metrics['Macro F1'].append(f1_score(y_true, y_pred, average='macro'))
        accum_metrics['Macro precision'].append(precision_score(y_true, y_pred, average='macro'))
        accum_metrics['Macro recall'].append(recall_score(y_true, y_pred, average='macro'))
        accum_metrics['F1 (Class 1)'].append(f1_score(y_true, y_pred, pos_label=1))
        accum_metrics['Precision (Class 1)'].append(precision_score(y_true, y_pred, pos_label=1))
        accum_metrics['Recall (Class 1)'].append(recall_score(y_true, y_pred, pos_label=1))

    # Compute mean of each metric
    mean_metrics = {key: np.mean(values) for key, values in accum_metrics.items()}
    mean_metrics['Dataset'] = filename
    mean_metrics['Chunks evaluated'] = n_chunks

    return mean_metrics


# Evaluate the model on multiple test files
results = []
file_list = ['Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag', 'Syn']

for filename in file_list:
    print(f"Processing {filename}")
    metrics =  test_file_order(features, scaler, autoencoder, loss, threshold, filename)
    results.append(metrics)

# Convert list of dicts into a DataFrame
metrics_df = pd.DataFrame(results)
# Melt the dataframe to long format for seaborn
melted_df = metrics_df.melt(id_vars='Dataset', var_name='metrics', value_name='value')

#Plot the metrics for each of the files
plt.figure(figsize=(18, 7))
ax = sns.barplot(data=melted_df, x='Dataset', y='value', hue='metrics') 
# Add metric value labels inside each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='center',rotation=90, fontsize=14, color='white'  )
plt.xticks(rotation=45, ha='right')
plt.title('Classification Metrics per Dataset')
ax.legend(loc='best', bbox_to_anchor=(1.02, 0.5), title='Metrics')
plt.tight_layout()
plt.show()

print(metrics_df)

#Compute the average metrics
averages = metrics_df.drop(columns=['Dataset']).mean()
print(averages)

#Save these results 
average_row = averages.to_frame().T  
average_row['Dataset'] = 'Overall Metrics'
metrics_df = pd.concat([metrics_df, average_row], ignore_index=True)
metrics_df.to_csv('results/autoencoder0_allfeatures.csv', index=False)







