# Higgs-Discovery

This project leverages the higgs dataset from the corresponding Kaggle competition 
to showcase a set of common machine learning tools from preprocessing to final
model evaluation and testing. 

## Requirements
1- python 3.4+
2- pandas
3- seaborn
4- matplotlib
5- scikit-learn
6- pickle

## Running the code

   The code base includes the data in the project files folder. The training and testing
data must be unzipped and moved to the Datasets folder. From there, the user can 
run  import_data.py to generate a pickle file of the original dataset. 

   Preprocessing.py can then be used to create all different versions of the original 
dataset but applying any of a series of preprocessing procedures like PCA, MI 
filtering, etc... The resulting datasets will be pickled in the Datasets folder.
   
   The main model creation and evaluation scripts are Model_Evaluation_and_Validation.py
and Model_Testing.py. The user can user the former to build crossvalidated models using 
any of the classifiers included in Classifiers.py. A name for the pickled model should be 
hardcoded so the model can be saved for testing. The latter script on the other hand will
produce a csv file in the format required by the Kaggle competition. The file can then be
submitted to Higgs Kaggle submission page to obtain the AMS score of the built model.
