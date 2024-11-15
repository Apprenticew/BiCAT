# BiCAT: An Effective Semi-supervised Deep Learning Model for BI-RADS-based Breast Cancer Screening in Asymptomatic Individuals

**Hong-Han Wang1, Zhi-Jie Chen2, Chong-Yang Xu1, Li Gao2, Zi-Mo Wang1, Guan-Lin Mo1,
Jia-Xiang Chen1, Wei-Chen Lin1, Jin-Peng Zhang1, Xiu-Li Dai2,
Hu Ding1*, Yong-Liang Zhang2*
**


**Abstract:** DEarly ultrasound screening for breast cancer aims to assess risk in asymptomatic individuals using the 
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

## Requirements

```python
pip install -r requirements.txt
```

## Quickstart

The dataset from Anhui Provincial Hospital Health Center used in this study is not publicly available.

The test code is for verification only

Due to excessive storage space, it is not possible to submit the training model, which means that the following code 
cannot be directly executed for testing. If necessary, please contact the author.

```python
python main.py --test
```

If you want to retrain the model, reference samples have been provided for reference only

Provide sample data for testing the repeatability of training only

```python
python main.py
```


## Citation

If you use our code, please cite us. 
