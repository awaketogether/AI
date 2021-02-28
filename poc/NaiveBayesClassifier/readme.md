# How to run the following POC

### How to run:
Using a raw data from an actigraphy process, run the following:
- (version >3.5 it's working) `python3 algorithm.py` (to see the confusion matrix with plotlib). *on a data file named `data.csv`*
- (only version 2.7) `python algorithm.py`(to save the model using jotlib and load it to use it in another data). *on a data file named `data_Patient_2.csv`*

### Python List package required
Package         Version
--------------- -------
cycler          0.10.0 // Do not sure it's relevant for this POC but it was in my list.
joblib          1.0.0  // Only works with python 2.7 (see following section).
kiwisolver      1.3.1  // Do not sure it's relevant for this POC but it was in my list.
matplotlib      2.0.0
numpy           1.19.5
pandas          1.2.1 // Do not sure it's relevant for this POC but it was in my list.
Pillow          8.1.0 // Do not sure it's relevant for this POC but it was in my list.
pip             20.3.3
pycairo         1.20.0 // Do not sure it's relevant for this POC but it was in my list.
PyGObject       3.38.0 // Do not sure it's relevant for this POC but it was in my list.
pyparsing       2.4.7 // Do not sure it's relevant for this POC but it was in my list.
python-dateutil 2.8.1 // Do not sure it's relevant for this POC but it was in my list.
pytz            2020.5 // Do not sure it's relevant for this POC but it was in my list.
scikit-learn    0.24.1
scipy           1.6.0 // Do not sure it's relevant for this POC but it was in my list.
seaborn         0.11.1 
setuptools      51.1.1 // Do not sure it's relevant for this POC but it was in my list.
six             1.15.0 // Do not sure it's relevant for this POC but it was in my list.
threadpoolctl   2.1.0 // Do not sure it's relevant for this POC but it was in my list.
wheel           0.36.2 // Do not sure it's relevant for this POC but it was in my list.

## POC Description
A Naive bayes classification algorithm implementation on Awake Data. Using the Gaussian Naive bayes (why? https://www.quora.com/What-is-the-difference-between-the-the-Gaussian-Bernoulli-Multinomial-and-the-regular-Naive-Bayes-algorithms).

## POC Report (Training model Explanations, Results (score, confusion matrix))
https://drive.google.com/file/d/1IurIkbacE8-3YIFflD_mF--BTdZxKcI_/view?usp=sharing

## Authors & Co-authors
- [Simon Provost](https://github.com/simonprovost)