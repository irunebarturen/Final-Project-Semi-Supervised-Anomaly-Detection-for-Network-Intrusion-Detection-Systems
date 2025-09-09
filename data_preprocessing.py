
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, VarianceThreshold
from sklearn.manifold import TSNE


#### DATA PREPROCESSING

## Count number of flows per class in each of the files
file_list = [
    'CSV_01_12/DrDoS_DNS', 'CSV_01_12/DrDoS_LDAP', 'CSV_01_12/DrDoS_MSSQL',
    'CSV_01_12/DrDoS_NetBIOS', 'CSV_01_12/DrDoS_NTP', 'CSV_01_12/DrDoS_SNMP',
    'CSV_01_12/DrDoS_SSDP', 'CSV_01_12/DrDoS_UDP', 'CSV_01_12/Syn', 'CSV_01_12/UDPLag',
    'CSV_03_11/LDAP', 'CSV_03_11/MSSQL', 'CSV_03_11/NetBIOS', 'CSV_03_11/Portmap', 
    'CSV_03_11/UDP', 'CSV_03_11/UDPLag'
]

chunksize = 10_000
base_path = "/datasets/CICDDoD2019/"

for filename in file_list:
    print(f"\nProcessing {filename}")
    path = os.path.join(base_path, filename + '.csv')

    label_counts = Counter()
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[' Label'], low_memory=False):
        label_counts.update(chunk[' Label'])

    print(label_counts)



## Save all benign flows from CSV_01_12 each file in a separate file that will be used for training
file_list = ['DrDoS_DNS','DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS','DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP' , 'Syn', 'UDPLag']
base_path = "/datasets/CICDDoD2019/CSV_01_12"

for i in file_list:
    path = os.path.join(base_path, i + '.csv')   
    data = pd.read_csv(path)
    data = data.sort_values([' Timestamp'])
    df_subset = data[data[' Label']== 'BENIGN']
    df_subset['File'] = i
    csv_file = 'benign_01_12.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)
    # Append to CSV (or create it if not exists)
    df_subset.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    print(i)



## Plot the benign data from CSV_01_12 we will use for training.
def plot_normal(df):
    df[' Timestamp'] = pd.to_datetime(df[' Timestamp'])
    df['Minute'] = df[' Timestamp'].dt.floor('min')
    volume = df.groupby('Minute').size()

    # Plot
    plt.figure(figsize=(12, 6))
    volume.plot(kind='area', stacked=True, alpha=0.8)#, color='skyblue')
    plt.title("Normal Traffic Volume Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Flows per Minute")
    plt.tight_layout()
    plt.show()

data_benign = pd.read_csv('/home/irune_barturen/final_project/final project/code/benign_01_12.csv')
plot_normal(data_benign)


#### FEATURE SELECTION

def feature_selection_composite(filename, label_col=' Label', top_k=10, seed=42):
    
    path = os.path.join(r"/datasets/CICDDoD2019/", filename + '.csv')   

    sample_frac = 0.15  # 20% random sample
    chunksize = 10_000
    sampled_chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        sampled_chunk = chunk.sample(frac=sample_frac, random_state = seed)
        sampled_chunks.append(sampled_chunk)
        

    data = pd.concat(sampled_chunks, ignore_index=True)

    # Drop unneeded columns
    drop_cols = ['Unnamed: 0', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    # Clean NaNs and infs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Separate features and target
    X = data.drop(columns=[label_col])
    y = (data[label_col] != 'BENIGN').astype(int)  # 0 = benign, 1 = attack

    # Variance threshold
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(X)
    dropped_features = X.columns[~vt.get_support()]    # Features dropped
    X = pd.DataFrame(X_vt, columns=X.columns[vt.get_support()])
    
    # Print dropped features
    if len(dropped_features) > 0:
        print("Dropped low-variance features:")
        for feature in dropped_features:
            print(f"  - {feature}")
    else:
        print("No features were dropped due to low variance.")

    # Supervised feature selection
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    mi_scores = mutual_info_classif(X, y, random_state=seed)
    f_scores, _ = f_classif(X, y)

    # Combine scores
    combined = (
        pd.DataFrame({
            'feature': X.columns,
            'rf': rf_importances,
            'mi': mi_scores,
            'f': f_scores
        })
        .set_index('feature')
    )

    # Normalize and average scores
    combined = combined / combined.max()
    combined['score'] = combined.mean(axis=1)

    # Select top-k features
    selected_features = combined['score'].sort_values(ascending=False).head(top_k).index.tolist()

    return selected_features, combined.sort_values('score', ascending=False)


feature_scores_list = []
file_list_train = [
    'CSV_01_12/DrDoS_DNS', 'CSV_01_12/DrDoS_LDAP', 'CSV_01_12/DrDoS_MSSQL',
    'CSV_01_12/DrDoS_NetBIOS', 'CSV_01_12/DrDoS_NTP', 'CSV_01_12/DrDoS_SNMP',
    'CSV_01_12/DrDoS_SSDP', 'CSV_01_12/DrDoS_UDP', 'CSV_01_12/Syn', 'CSV_01_12/UDPLag',
]

for filename in file_list_train:
    print(f"\nProcessing {filename}")
    selected_features, scores = feature_selection_composite(filename, top_k=10)
    top_scores = scores.loc[selected_features]

    # Store as a dictionary: {feature: score}
    feature_scores_list.append(top_scores.to_dict())


top_features_df = pd.DataFrame(feature_scores_list, index=file_list_train)

# Display the top 10 features per dataset
print("Top features per dataset:")
print(top_features_df)

# Save the updated DataFrame to CSV
top_features_df.to_csv('feature_selection/features_importance', index=False)


scores=top_features_df['score']
features=[]
for d in scores:
    features.extend(d.keys())
#Total number of features collected (with duplicates)
print(len(features))
# Print the number of unique features selected across all datasets
set_features = set(features)
print(len(set_features))

#### CREATE NEW DATASETS FOR TESTING (CSV_03/11) WHERE JUST AN ATTACK TYPE APPEAR IN EACH DATASET
#Datasets we need to separate: CSV 03 11/LDAP, CSV 03 11/MSSQL, CSV 03 11/UDP, CSV 03 11/UDPLag
#1. filename = 'CSV_03_11/LDAP' attack1_label = 'NetBIOS' attack2_label = 'LDAP'
#2. filename = 'CSV_03_11/MSSQL' attack1_label = 'LDAP'  attack2_label = 'MSSQL'
#3. filename = 'CSV_03_11/UDP' attack1_label = 'MSQL'  attack2_label = 'UDP'
#Be careful with the next ones, we need to separate UDPLag in three datasets
#4. filename = 'CSV_03_11/UDPLag' attack1_label = 'UDP'  attack2_label = 'UDPLag'
#5. filename = 'UDPLag_new_copy' attack1_label = 'UDPLag'  attack2_label = 'Syn' 

filename = 'CSV_03_11/LDAP' 
attack1_label = 'NetBIOS' 
attack2_label = 'LDAP'
output_dir = "/datasets/CICDDoD2019_test_new"
os.makedirs(output_dir, exist_ok=True)

path = os.path.join("/datasets/CICDDoD2019/", filename + '.csv')

chunksize = 50_000
part1_rows = []
attack2_found = False
part2_path = os.path.join(output_dir, f"{attack2_label}_new.csv")
part1_path = os.path.join(output_dir, f"{attack1_label}_new.csv")

with pd.read_csv(path, chunksize=chunksize, low_memory=False) as reader:
    for chunk in reader:
        # Ensure timestamp ordering within each chunk
        chunk = chunk.sort_values(' Timestamp').reset_index(drop=True)
        
        if not attack2_found:
            # Check if Attack-2 appears in this chunk
            if attack2_label in chunk[' Label'].values:
                first_idx = chunk[chunk[' Label'] == attack2_label].index[0]
                part1_chunk = chunk.iloc[:first_idx]
                part2_chunk = chunk.iloc[first_idx:]
                
                # Store part1 and write part2
                part1_rows.append(part1_chunk)
                part2_chunk.to_csv(part2_path, index=False, mode='w', header=True)
                attack2_found = True
            else:
                part1_rows.append(chunk)
        else:
            # Already found Attack-2, keep writing remaining chunks to part2
            chunk.to_csv(part2_path, index=False, mode='a', header=False)

# Combine and save part1
part1_df = pd.concat(part1_rows, ignore_index=True)
part1_df.to_csv(part1_path, index=False, mode='a', header = True)


## Count number of flows per class in each of the new files
file_list = [
    'Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag', 'Syn'
]

chunksize = 10_000
base_path = "/datasets/CICDDoD2019_test_new"

for filename in file_list:
    print(f"\nProcessing {filename}")
    path = os.path.join(base_path, filename + '.csv')

    label_counts = Counter()
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[' Label'], low_memory=False):
        label_counts.update(chunk[' Label'])

    print(label_counts)


##Compute the first and last timestamp for each of the files
#For new files
base_path = "/datasets/CICDDoD2019_test_new"
file_list = ['Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag', 'Syn']

#For original files
base_path = "/datasets/CICDDoD2019"
file_list = [
    'CSV_01_12/DrDoS_DNS', 'CSV_01_12/DrDoS_LDAP', 'CSV_01_12/DrDoS_MSSQL',
    'CSV_01_12/DrDoS_NetBIOS', 'CSV_01_12/DrDoS_NTP', 'CSV_01_12/DrDoS_SNMP',
    'CSV_01_12/DrDoS_SSDP', 'CSV_01_12/DrDoS_UDP', 'CSV_01_12/Syn', 'CSV_01_12/UDPLag',
    'CSV_03_11/LDAP', 'CSV_03_11/MSSQL', 'CSV_03_11/NetBIOS', 'CSV_03_11/Portmap', 
    'CSV_03_11/UDP', 'CSV_03_11/UDPLag'
]

chunksize = 10_000

for filename in file_list:
    print(f"\nProcessing {filename}")
    path = os.path.join(base_path, filename + '.csv')

    first_timestamp = None
    last_timestamp = None

    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        # First chunk → take first row
        if i == 0:
            first_timestamp = chunk.iloc[0][' Timestamp']  # change 'Time' to your actual column name

        # Always take the last row of the current chunk
        last_timestamp = chunk.iloc[-1][' Timestamp']

    print(f"First timestep: {first_timestamp}")
    print(f"Last timestep: {last_timestamp}")


#### VISUALIZE THE DATASETS USING t-SNE
#31 features selected above
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

data_benign_features = pd.read_csv('benign_01_12.csv', usecols=features)
data_benign_features.replace([np.inf, -np.inf], np.nan, inplace=True)
data_benign_features.dropna(inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_benign_features)

# Loads a large CSV in chunks and returns a balanced sample of benign (%70 proportion) and attack(5,000) rows using random sampling.
def sample_mixed_rows_from_csv(path, usecols, benign_label='BENIGN', benign_frac=0.7, attack_sample_size=5000, chunksize=100_000, seed=42):
    rng = np.random.default_rng(seed)
    benign_rows = []
    attack_reservoir = []
    total_attacks_seen = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False, nrows=3_500_000):
        benign_chunk = chunk[chunk[' Label'] == benign_label]
        attack_chunk = chunk[chunk[' Label'] != benign_label]

        # Sample 70% of the benign data
        benign_sample = benign_chunk.sample(frac=benign_frac, random_state=seed)
        benign_rows.append(benign_sample)

        # Reservoir sampling for attack rows
        for _, row in attack_chunk.iterrows():
            total_attacks_seen += 1
            if len(attack_reservoir) < attack_sample_size:
                attack_reservoir.append(row)
            else:
                j = rng.integers(0, total_attacks_seen)
                if j < attack_sample_size:
                    attack_reservoir[j] = row

    # Combine and return as DataFrame
    benign_df = pd.concat(benign_rows, ignore_index=True)
    attack_df = pd.DataFrame(attack_reservoir)
    return pd.concat([benign_df, attack_df], ignore_index=True)

# Loads sampled data, applies t-SNE for dimensionality reduction, and visualizes the data distribution by class.
def tsne_file(features, filename, seed=42):
    
    path =  os.path.join(r"/datasets/CICDDoD2019_test_new/", filename + '.csv') 
    data = sample_mixed_rows_from_csv(path, features + [' Label'],  attack_sample_size=10000, chunksize=100_000, seed=42)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    X = data.drop(columns=[' Label'])
    y_true = (data[' Label'] != 'BENIGN').astype(int)  # Binary classification: 1 if attack, 0 if benign
    
    X_scaled = scaler.transform(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    X_embedded = tsne.fit_transform(X_scaled)

    # Plot t-SNE
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1],hue=y_true.map({0: 'Benign', 1: filename + ' Attack'}), palette={'Benign': 'blue', filename + ' Attack': 'red'}, alpha=0.6)
    plt.title(f't-SNE Visualization for {filename}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.show()

file_list = [
    'Syn', 'Portmap', 'NetBIOS', 'LDAP', 'MSSQL', 'UDP', 'UDPLag'
]
file_list = [
  'MSSQL'
]
for filename in file_list:
    print(f"Processing {filename}")
    metrics = tsne_file(features, filename)
  
