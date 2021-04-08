# How to run the following POC

### How to run:
Using a raw data from an actigraphy process, run the following:
- (version >3.5 it's working) `python3 algorithm.py` 
- (only version 2.7) `python algorithm.py`

### Python List package required

Package
---------------
pandas==1.2.1
seaborn==0.11.1
matplotlib==3.3.4
numpy==1.20.0
scikit_learn==0.24.1


## POC Description
The use of K-NN or instance-based learning is very powerful since it computes the K nearest neighbours and groups them together if they pass a certain threshold. 
Moreover, our data at Awake is perfectly suited to the sense of K-NN. However, as described in the drive report below, there is an overfitting behaviour because we have limitations. Additional research would be needed.

## POC Report (Training model Explanations, Results (score, confusion matrix))
https://docs.google.com/document/d/1m8nEQEKEuuSknT7f_dSgCPGxt4ElzR8ph_4AI1DCDv4/edit?usp=sharing

## Authors & Co-authors
- [Simon Provost](https://github.com/simonprovost)