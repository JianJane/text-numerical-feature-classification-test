#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:35:09 2019

automated training script for McarthyFintch A.I juniro engineer tech test

@author: jbi
"""
import pandas as pd
import re
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class documentClassifier:
    
    def __init__(self, dataFilePath):
        
        self.path_name= dataFilePath+'t-data.csv'
        
        self._data_orig= pd.read_csv(self.path_name, sep=',')
        
        self._data_inprocess=self._data_orig
        
        self._preprocessed_data= 0
        
        self._new_data= 0
        
        self._new_data_processed=0
        
        
        self.vectorizer=0
        
        self.encoder=0
        
        self.corpus_legal=0
        self.corpus_admin=0
        self.corpus_hr=0
        self.corpus_sales=0
        self.corpus_other=0
        
        self.X=0
        
        self.y=0
        
        
        self.clf_trained=0
        
        self.accuracy_scores_train=[]
        
        self.accuracy_scores_test=[]
        
        self.pred_result=0
        
        self.pred_proba=0
        
        
    def set_preprocessed_data(self, preprocessed_data):
         
        self._preprocessed_data=preprocessed_data
        
    def set_new_data_point(self, new_data):
        
        self._new_data=new_data
            
    def transform_new_data(self):
        
        """
        function only activated for 'predict' mode
        it appends the new data onto the training set, enable new data to be pre-processed as the training data
        
        """
    
        colnames=self._data_orig.columns
        
        n_data=self._new_data.split(',')
        
        for i in range(3):
            str_v=n_data[i]
            n_data[i]=float(str_v)
        
        n_data.append('legal')
                
        n_D=pd.DataFrame(n_data)
        n_D=n_D.T
        
        n_D.columns=colnames
        
        self._data_inprocess=self._data_inprocess.append(n_D).reset_index(drop=True)
        
        
    def columns_transformations(self):
        """
        transforming and combining some of the non-text columns into new features which might optimise the model 
        
        1. 'wordCount' and 'pagesCount' are combined into 'wordCountPage' or word per page
        2. 'wordCountPage' is StandardScaled into 'wordCountPage_scaled'
        3. 'author' is processed using Regex into 'author_gender'
        4. 'createdDate' is processed into 5 different columns, 'weekday'(numerical), 'workday'(boolean), 'time', 'date', 'workingHours'
        
        """        
        # cerating word_per_page column
        data=self._data_inprocess
        
        data['wordCountPage']= data['wordCount']/data['pagesCount']
        data=data.drop(['pagesCount','wordCount', 'fileSize'], axis=1)
        
        # feature scaling 'wordCountPage'
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data['wordCountPage']).reshape(-1,1))
        data['wordCountPage_scaled']=scaled_data
        
                
        # transforming author column into gender column
        def split_title(title):
            
            """
            spliting the name titles to identify the gender of the author
            this function is applied to the 'author' column
            
            """
            
            title=title.lower()
            title_l=re.sub(r"\.", " ", title)
            title_l=re.split(r"\s", title_l)
            if title_l[0] in ['mister', 'mr']:
                gender=1
            else:
                gender=0
            return(gender)
            
        data['author_gender']=data['author'].apply(split_title)
        
        def parse_datetime_date( datetime_str):
            """
            parsing datetime string from the feature space into datetime object for daytime object extraction
            
            """
            
            datetime_str=datetime_str.lower()
            datetime_str=re.sub(r"t|(.000z)", " ", datetime_str)
        
            datetime_pattern='%Y-%m-%d %H:%M:%S '
            date_time_obj = datetime.datetime.strptime(datetime_str, datetime_pattern)
                        
            return(date_time_obj)
            
        
        data['datetime']=data['createdDate'].apply(parse_datetime_date)
        data['weekday']=data['datetime'].apply(lambda x: x.weekday())
        data['workday']=data['weekday'].apply(lambda x: 1 if x<=4 else 0)
        data['time']=data['datetime'].apply(lambda x: x.time())
        data['date']=data['datetime'].apply(lambda x: x.date())
        
        # filtering working hours Boolean
        
        start_time=datetime.datetime.strptime('08:00:00', '%H:%M:%S').time()
        end_time=datetime.datetime.strptime('18:00:00', '%H:%M:%S').time()
        
        data['workingHours']=(data['time']<end_time)&(data['time']>start_time)
        data['workingHours']=data['workingHours'].apply(int)

        self._data_inprocess=data
        
            
    def title_column_fEngineering(self):
        
        """
        title column text feature engineering into word of vector norm column using bag of words for different categories
        
        
        """
        
        data=self._data_inprocess
        
        def clean_text(text):
            """
            using regex to clean the title text, this function is embeded in the CountVectorizer as below
            
            """
            patterns= r"\W+|-|_|[0-9]+"
            corpus_clean=re.sub(patterns, ' ', text)
            return(corpus_clean)
        
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', preprocessor=clean_text)
    
        
        corpus_legal=[sent for sent in data.loc[data['category']=='legal', 'title']]
        corpus_other=[sent for sent in data.loc[data['category']=='other', 'title']]
        corpus_sales=[sent for sent in data.loc[data['category']=='sales', 'title']]
        corpus_hr=[sent for sent in data.loc[data['category']=='hr', 'title']]
        corpus_admin=[sent for sent in data.loc[data['category']=='admin', 'title']]
    
        bow_other=vectorizer.fit(corpus_other)
        bow_legal=vectorizer.fit(corpus_legal)
        bow_sales=vectorizer.fit(corpus_sales)
        bow_hr=vectorizer.fit(corpus_hr)
        bow_admin=vectorizer.fit(corpus_admin)
    
        
        def wordVector_sum(row, bow_cat):
            
            """
            function transforms the 'title' using the bag of words of one of the categories, 
            then calculate its vector sum as norm 
            """
            
            titleVector_sum=bow_cat.transform([row['title']]).toarray().sum()
            
            return(titleVector_sum)
        
        data['titleVectorSum_legal']=data.apply(lambda x: wordVector_sum(x, bow_legal), axis=1)
        data['titleVectorSum_other']=data.apply(lambda x: wordVector_sum(x, bow_other), axis=1)
        data['titleVectorSum_sales']=data.apply(lambda x: wordVector_sum(x, bow_sales), axis=1)
        data['titleVectorSum_admin']=data.apply(lambda x: wordVector_sum(x, bow_admin), axis=1)
        data['titleVectorSum_hr']=data.apply(lambda x: wordVector_sum(x, bow_hr), axis=1)
        
        data=data.drop(['weekday','time','date','datetime','createdDate','wordCountPage','title','author'],axis=1)
        
        
        self._new_data_processed=data.iloc[-1,:].drop(labels=['category'])
        self._data_inprocess=data[:-1]
        
    def encode_Xy(self):
        """`
        encoding label column into intergers
        updating X processed feature matrix
        updating y encoded label array
        """
        
        le = LabelEncoder()
        
        # apply le on categorical feature columns
        self._data_inprocess['category_encode'] = le.fit_transform(self._data_inprocess['category'])
        
        self.encoder=le
        
        self.y = np.array(self._data_inprocess['category_encode'])
        self.X = self._data_inprocess.drop(['category_encode', 'category'], axis=1)
        
    def stratified_cv(self, classifier, nfold=10):
        
        """
        using StratifiedKFold module to perform stratified train_test split 
        using stratified train and test set for cross validation
        the accuracy scores on training and test sets are saved in their respetive arrays as class variables
        
        """
    
        X=self.X
        y=self.y
        
        kf = StratifiedKFold(n_splits=nfold, random_state=None, shuffle=False)
       
        accuracy_scores_train=[]
        accuracy_scores_test=[]
        
        for train_index, test_index in kf.split(X, y):
            
            clf = classifier
        
            X_train=X.iloc[train_index,:]
            y_train=y[train_index]
            X_test=X.iloc[test_index,:]
            y_test=y[test_index]
            
            self.clf_trained=clf.fit(X_train, y_train)
        
            predictions_test = clf.predict(X_test)
            predictions_train = clf.predict(X_train)
            
            accuracy_scores_train.append(accuracy_score(predictions_train, y_train))
            accuracy_scores_test.append(accuracy_score(predictions_test, y_test))
            
            self.accuracy_scores_train=accuracy_scores_train
            self.accuracy_scores_test=accuracy_scores_test
            
    
    def show_accuracy_scores(self):
        """
        print the accuracy scores for model validated on training set
        print the accuracy scores for model validated on testing set
        """
        
        print('training set 10 fold cross validation scores: {}'.format(self.accuracy_scores_train))
        print('test set 10 fold cross validation scores scores: {}'.format(self.accuracy_scores_test))               

    def predict(self):
        
        """
        using trained model for prediction
        converting the numerical result to a categorical output
        predition output saved as class varialbe
        """
        
        pred_result=self.clf_trained.predict([self._new_data_processed])
        
        self.pred_result=self.encoder.inverse_transform(pred_result)
        
    def show_prediction(self):
        
        print('prediction : {}'.format(self.pred_result))
#        print('prediction probability : {}'.format(self.pred_proba))
    
def main():
    # user input file paths
    dataFilePath=input("Enter labeled data file path: ")
    
    mode=input("mode=")
    
    dCls=documentClassifier(dataFilePath)
    
    ###### create classifier object ###########
    
    from sklearn.ensemble import RandomForestClassifier

    forest_clf=RandomForestClassifier(n_estimators=30)        
    
    
    if mode=='train':
    
        dCls.columns_transformations()
        dCls.title_column_fEngineering()
        dCls.encode_Xy()
        
        dCls.stratified_cv(forest_clf)
        dCls.show_accuracy_scores()
        
        
    if mode=='predict':
        
        new_data=input("input=")
        
        dCls.set_new_data_point(new_data)
        dCls.transform_new_data()
        
        dCls.columns_transformations()
        dCls.title_column_fEngineering()
        dCls.encode_Xy()
        
        dCls.stratified_cv(forest_clf)
        dCls.predict()
        dCls.show_prediction()
        
    # return object handle to user    
    return(dCls)
    

if __name__ == "__main__":
    main()        
    
