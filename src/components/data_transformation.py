import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomeException
from src.logger import logging

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer(self):

        try:
            cat_cols=['MARRIAGE','EDUCATION']
            num_cols=['LIMIT_BAL', 'SEX', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
                      'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                      'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                      'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                      ]
                        
                                   
            num_pipeline=Pipeline(
                steps=[
                 ('scaler',StandardScaler())
                 ]
                   )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('onehotencoder',OneHotEncoder(sparse=False)),
                ('scaler',StandardScaler())
                 ] 
                    )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
                ]
                )
            logging.info('numerical columns standard scaling completed')
            logging.info('categorical columns encoding completed')

            return preprocessor
            
        except Exception as e:
            raise CustomeException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        logging.info('Data transformation has started')
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            preprocessor_obj=self.get_data_transformer()

            target_column_name='default payment next month'

            input_feature_train_df=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_data[target_column_name]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )


        except Exception as e:
            raise CustomeException(e,sys)