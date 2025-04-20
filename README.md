Final Dataset found at: https://drive.google.com/file/d/1eYxxgGCjZO77G_avOLcWQ7BEkzPx3Dui/view?usp=drive_link
python version: 3.11.4 [Project uses Torch, which may not be supported by some of the newer versions]

Datasets used in this project:


# How to try

If you don't want to modify local versions please use virtual env

[OPTIONAL]
```
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

```
pip install -r requirements.txt
```
After installation, run this command. It should open the interface on local host.

```
streamlit run app.py
```


# Testing / Retraining

When running on google collab, download and copy this zip into your google drive to run cds_project_Patching.ipynb and CDS_SVM_+_VLM.ipynb. Additionally for CDS_SVM_+_VLM.ipynb also paste the .csv files into content of google collab. Alternatively you can modify the filepaths.

# Results
## Without Validation [best_vlm_model_no_validation.pth]
### Classification Report

| Class                     | Precision | Recall | F1-score | Support |
|--------------------------|-----------|--------|----------|---------|
| Benign                   | 0.98      | 0.92   | 0.95     | 71      |
| [Malignant] Pre-B        | 0.99      | 0.99   | 0.99     | 134     |
| [Malignant] Pro-B        | 0.99      | 1.00   | 1.00     | 112     |
| [Malignant] early Pre-B  | 0.96      | 0.99   | 0.97     | 137     |
|                          |           |        |          |         |
| **Accuracy**             |           |        | **0.98** | **454** |
| **Macro avg**            | 0.98      | 0.97   | 0.98     | 454     |
| **Weighted avg**         | 0.98      | 0.98   | 0.98     | 454     |

## With validation [best_vlm_model.pth]
### Classification Report 

| Class                     | Precision | Recall | F1-score | Support |
|--------------------------|-----------|--------|----------|---------|
| Benign                   | 0.97      | 0.95   | 0.96     | 78      |
| [Malignant] Pre-B        | 0.99      | 0.99   | 0.99     | 144     |
| [Malignant] Pro-B        | 0.99      | 0.99   | 0.99     | 120     |
| [Malignant] early Pre-B  | 0.98      | 0.99   | 0.98     | 148     |
|                          |           |        |          |         |
| **Accuracy**             |           |        | **0.98** | **490** |
| **Macro avg**            | 0.98      | 0.98   | 0.98     | 490     |
| **Weighted avg**         | 0.98      | 0.98   | 0.98     | 490     |

### Training and Validation Loss per Epoch

| Epoch | Train Loss | Val Loss | Best Model Saved |
|-------|------------|----------|------------------|
| 1     | 1.1526     | 1.0073   | ✅                |
| 2     | 0.9149     | 0.8087   | ✅                |
| 3     | 0.7401     | 0.6590   | ✅                |
| 4     | 0.6081     | 0.5475   | ✅                |
| 5     | 0.5053     | 0.4623   | ✅                |
| 6     | 0.4340     | 0.4041   | ✅                |
| 7     | 0.3775     | 0.3556   | ✅                |
| 8     | 0.3370     | 0.3176   | ✅                |
| 9     | 0.3010     | 0.2852   | ✅                |
| 10    | 0.2720     | 0.2675   | ✅                |
| 11    | 0.2504     | 0.2382   | ✅                |
| 12    | 0.2297     | 0.2232   | ✅                |
| 13    | 0.2108     | 0.2042   | ✅                |
| 14    | 0.1979     | 0.1924   | ✅                |
| 15    | 0.1806     | 0.1790   | ✅                |
| 16    | 0.1679     | 0.1718   | ✅                |
| 17    | 0.1591     | 0.1536   | ✅                |
| 18    | 0.1475     | 0.1475   | ✅                |
| 19    | 0.1407     | 0.1405   | ✅                |
| 20    | 0.1347     | 0.1437   |                  |
| 21    | 0.1250     | 0.1221   | ✅                |
| 22    | 0.1176     | 0.1151   | ✅                |
| 23    | 0.1117     | 0.1105   | ✅                |
| 24    | 0.1075     | 0.1049   | ✅                |
| 25    | 0.1024     | 0.1000   | ✅                |
| 26    | 0.0967     | 0.0948   | ✅                |
| 27    | 0.0942     | 0.0904   | ✅                |
| 28    | 0.0908     | 0.1027   |                  |
| 29    | 0.0871     | 0.0863   | ✅                |
| 30    | 0.0824     | 0.0802   | ✅                |


## Training Process 
Blood Spread Image > Cell Patching [Main:fn_patching]  > Individual Cells to processs + Count of total White Blood cells

Individual Cell > [efficientnetv2_leukemia.keras] > Preict possibility of cancer
Individual Cell > [inceptionv3_leukemia.keras] > Preict possibility of cancer 

Cells > Main [fn_patching] > Mean, Max, Standard Deviation

X:{Count, Mean, Max, Standard Deviation, Blood Spread Image}, y:{Diagnosis} > VLM > Model


