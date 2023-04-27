import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomeException

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df=pd.read_csv(r'notebook\data\finalcreditcard.csv')

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomeException(e,sys)   
        

