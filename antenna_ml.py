import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Show all columns in output
pd.set_option('display.max_columns', None)

# Step 1: Load your CSV
df = pd.read_csv('antenna_data.csv')

# Step 2: Clean column names
df.columns = [col.strip() for col in df.columns]

# Step 3: Encode substrate type
le = LabelEncoder()
df['Substrate_encoded'] = le.fit_transform(df['Substrate'])

# Step 4: Select features and targets
X = df[['Substrate Thickness (mm)', 'Dielectric Constant', 'Substrate_encoded']]
y_return_loss = df['Return Loss (dB)']
y_vswr = df['VSWR']

# Step 5: Train/test split
X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(X, y_return_loss, test_size=0.3, random_state=42)
X_train_vswr, X_test_vswr, y_train_vswr, y_test_vswr = train_test_split(X, y_vswr, test_size=0.3, random_state=42)

# Step 6: Train regression models
model_rl = RandomForestRegressor(n_estimators=100, random_state=42)
model_rl.fit(X_train_rl, y_train_rl)

model_vswr = RandomForestRegressor(n_estimators=100, random_state=42)
model_vswr.fit(X_train_vswr, y_train_vswr)

# Step 7: Predict for all rows in your CSV file
df['Predicted Return Loss (dB)'] = model_rl.predict(X)
df['Predicted VSWR'] = model_vswr.predict(X)

# Step 8: Show results
print(df)

# Step 9: Save predictions for your paper/report
df.to_csv('antenna_ml_results.csv', index=False)

import matplotlib.pyplot as plt

# (Assume your DataFrame df already contains 'Substrate', 'Substrate Thickness (mm)',
# 'Predicted Return Loss (dB)', and 'Predicted VSWR' columns)

# Plot Predicted Return Loss vs Thickness
plt.figure(figsize=(8,5))
for substrate in df['Substrate'].unique():
    subset = df[df['Substrate'] == substrate]
    plt.plot(subset['Substrate Thickness (mm)'], subset['Predicted Return Loss (dB)'], 
             marker='o', label=substrate)
plt.title('ML-Predicted Return Loss vs. Substrate Thickness')
plt.xlabel('Substrate Thickness (mm)')
plt.ylabel('Predicted Return Loss (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Predicted VSWR vs Thickness
plt.figure(figsize=(8,5))
for substrate in df['Substrate'].unique():
    subset = df[df['Substrate'] == substrate]
    plt.plot(subset['Substrate Thickness (mm)'], subset['Predicted VSWR'], 
             marker='s', label=substrate)
plt.title('ML-Predicted VSWR vs. Substrate Thickness')
plt.xlabel('Substrate Thickness (mm)')
plt.ylabel('Predicted VSWR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Ensure Substrate is treated as a string
df['Substrate'] = df['Substrate'].astype(str)

# Plot Predicted VSWR vs Thickness
plt.figure(figsize=(8,5))
for substrate in df['Substrate'].unique():
    subset = df[df['Substrate'] == substrate]
    plt.plot(subset['Substrate Thickness (mm)'], subset['Predicted VSWR'], 
             marker='s', label=substrate)
plt.title('ML-Predicted VSWR vs. Substrate Thickness')
plt.xlabel('Substrate Thickness (mm)')
plt.ylabel('Predicted VSWR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

# Generate fine grid: 0.8, 0.9, 1.0, ..., 3.2 mm
fine_thickness = np.arange(0.8, 3.21, 0.1)

substrate_types = ['FR-4', 'Rogers RT']
dielectric = {'FR-4': 4.4, 'Rogers RT': 2.2}

fine_samples = []
for sub in substrate_types:
    for thick in fine_thickness:
        fine_samples.append([thick, dielectric[sub], le.transform([sub])[0], sub])

fine_df = pd.DataFrame(fine_samples, columns=['Substrate Thickness (mm)', 'Dielectric Constant', 'Substrate_encoded', 'Substrate'])

# Predict with ML model
fine_df['Predicted Return Loss (dB)'] = model_rl.predict(fine_df[['Substrate Thickness (mm)', 'Dielectric Constant', 'Substrate_encoded']])
fine_df['Predicted VSWR'] = model_vswr.predict(fine_df[['Substrate Thickness (mm)', 'Dielectric Constant', 'Substrate_encoded']])

print(fine_df)

# Plot for publication
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
for substrate in fine_df['Substrate'].unique():
    subset = fine_df[fine_df['Substrate'] == substrate]
    plt.plot(subset['Substrate Thickness (mm)'], subset['Predicted Return Loss (dB)'], label=f'{substrate} (ML, 0.1mm step)')
plt.title('ML Interpolated Return Loss vs. Thickness (0.1 mm step)')
plt.xlabel('Substrate Thickness (mm)')
plt.ylabel('Predicted Return Loss (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
for substrate in fine_df['Substrate'].unique():
    subset = fine_df[fine_df['Substrate'] == substrate]
    plt.plot(subset['Substrate Thickness (mm)'], subset['Predicted VSWR'], label=f'{substrate} (ML, 0.1mm step)')
plt.title('ML Interpolated VSWR vs. Thickness (0.1 mm step)')
plt.xlabel('Substrate Thickness (mm)')
plt.ylabel('Predicted VSWR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


