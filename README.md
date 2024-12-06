# BiCAT: An Effective Semi-supervised Deep Learning Model for BI-RADS-based Breast Cancer Screening in Asymptomatic Individuals

**
**


**Abstract:** Early ultrasound screening for breast cancer aims to assess risk in asymptomatic individuals using the 
standardized Breast Imaging Reporting and Data System (BI-RADS) atlas, enabling more appropriate management recommendations. 
Despite the transformative role of deep learning, the progress in medicine is often hindered by limited labeled training 
data and a large imbalance between benign and malignant cases (especially for asymptomatic individuals). In this article, 
we present an effective machine learning method,  BiCAT, for BI-RADS based Classification in Asymptomatic populations by 
a Two-stage deep learning model. By analyzing over 120,000 unlabeled and 1,000 labeled breast ultrasound images, BiCAT 
achieved an accuracy of 79.7% and a positive predictive value (PPV) of 13.6% in asymptomatic populations. The model’s 
performance was further validated on the cohorts comprising over 4,000 breast ultrasound images from multiple centers 
and time periods. In two independent reader studies involving 10 experienced radiologists from different institutions, 
BiCAT outperformed nearly every radiologist across BI-RADS categories 1-4. Compared to these radiologists, it reduced 
the false positives by half and increased the sensitivities by 9-19% in different modes. By providing visual classification 
heatmaps and clear confidence levels, BiCAT can effectively support radiologists in clinic applications. This approach 
demonstrates strong potential to enhance large-scale breast cancer ultrasound screening for asymptomatic populations.

## 1.System requirements

### All software dependencies and operating systems (including version numbers)
platform: linux-64

python3.8
### Versions the software has been tested on
Only this version
### Any required non-standard hardware
No
## 2.Installation guide
### Instructions
```python
conda create -n <your env name> python=3.8
pip install -r requirements.txt
```

### Typical install time on a "normal" desktop computer
20mins
## 3.Demo
### Instructions to run on data
The dataset from Anhui Provincial Hospital Health Center used in this study is not publicly available.

The test code is for verification only

```python
python main.py --test
```
Provide sample data for testing the repeatability of training only. 

The ```BiCAT\data1\Retrospective_train_cohort``` folder contains labeled training data, 
```BiCAT\data1\Retrospective_test_cohort``` contains labeled testing data, 
and ```BiCAT\unlabeled``` contains unlabeled training data.

```python
python main.py
```
### Expected output
Train:

Only example data and code are provided for proper execution, without considering the correctness of the output results.

Test:
```
result/
├── Prospective_test_cohort/  *This folder contains all the test data.*
│   ├── pred/  *Contains subfolders for predictions categorized by BI-RADS 1 to BI-RADS 4.*
│   │   ├── 1/  *BI-RADS 1*
│   │   │   ├── True/  *Correct predictions.*
│   │   │   │   ├── <True BI-RADS category>_filename.png
│   │   │   │   └── <Confidence level>_<True BI-RADS category>_filename.png
│   │   │   └── False/  *Incorrect predictions.*
│   │   │       ├── <True BI-RADS category>_filename.png
│   │   │       └── <Confidence level>_<True BI-RADS category>_filename.png
│   │   ├── 2/  *BI-RADS 2 (structure identical to BI-RADS 1).*
│   │   ├── 3/  *BI-RADS 3 (structure identical to BI-RADS 1).*
│   │   └── 4/  *BI-RADS 4 (structure identical to BI-RADS 1).*
```
### Expected run time for demo on a "normal" desktop computer
Test:20s

## 4.Instructions for use
### How to run the software on your data
You can store any ultrasound images in the ```BiCAT/data1/Prospective_test_cohort/1-4``` folders in the order of 
BI-RADS 1 to 4, then run 
```python
python main.py --test
```
to view the results in the ```result``` directory.

## License
This project is covered under the GNU General Public License

