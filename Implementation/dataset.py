import pandas as pd
import numpy as np
import os
import wget 
import datetime

class DataSet():
    def __init__(self):
        self.data_path = "/data/"
        self.url_dataset = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments.csv"
        self.col_names = {"col_id_product": "asin",
             "col_id_reviewer": "reviewerID",
             "col_rating": "overall",
             "col_unix_time": "unixReviewTime",
             "col_timestamp": "timestamp",
             "col_year": "year"}
        self.min = [0,0]
        self.csv_filename = ""
    
    def readDataSet(self, path, min_reviews, min_usuarios, dataset, nrows=None):

        self.min = [min_reviews, min_usuarios]
        if dataset != "movie lens":
            self.csv_filename = str(path + self.data_path) + "interactions_minR" + str(min_reviews) + "_minU" + str(min_usuarios) + ".csv"
            if not os.path.exists(self.csv_filename):
                self.create_df()
        else:
            self.csv_filename = str(path + self.data_path) + "interactions_movie_lens.csv"

        if nrows == None:
            df = pd.read_csv(self.csv_filename)
        else:
            df = pd.read_csv(self.csv_filename, nrows=nrows)
        return df
    
    def treat_dataset_src(self, df,info=False):

        # Type conversion
        df[self.col_names["col_rating"]] = pd.to_numeric(df[self.col_names["col_rating"]].replace(',','', regex=True))
        df[self.col_names["col_timestamp"]]=df[self.col_names["col_unix_time"]].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        df[self.col_names["col_year"]]= pd.to_datetime(df[self.col_names["col_timestamp"]]).dt.year

        # Duplicates
        df_duplicates = df[[self.col_names["col_id_reviewer"],self.col_names["col_id_product"], self.col_names["col_unix_time"]]].sort_values(by=[self.col_names["col_unix_time"]], ascending=False)
        df = df.drop(df_duplicates[df_duplicates[[self.col_names["col_id_reviewer"],self.col_names["col_id_product"]]].duplicated()][self.col_names["col_id_reviewer"]].index.values.tolist())

        productos_a_eliminar=[1]
        clientes_a_eliminar=[1]
        iteracion = 1
        while (len(productos_a_eliminar)!=0)and(len(clientes_a_eliminar)!=0):

            # Minimum number of users who have purchased the product products
            aux=df.groupby([self.col_names["col_id_product"]])[self.col_names["col_id_reviewer"]].count().reset_index()
            aux2=aux[aux[self.col_names["col_id_reviewer"]]<self.min[1]].reset_index() # usuarios a eliminar
            aux=aux[aux[self.col_names["col_id_reviewer"]]>=self.min[1]].reset_index() # usuarios a conservar
            productos=aux[self.col_names["col_id_product"]]
            df=df[df[self.col_names["col_id_product"]].isin(productos)]

            productos_a_eliminar=aux2[self.col_names["col_id_product"]]

            # Products selection with more than X reviews
            aux=df.groupby([self.col_names["col_id_reviewer"]])[self.col_names["col_rating"]].count().reset_index()
            aux2=aux[aux[self.col_names["col_rating"]]<self.min[0]].reset_index()
            aux=aux[aux[self.col_names["col_rating"]]>=self.min[0]].reset_index()
            clientes=aux[self.col_names["col_id_reviewer"]]
            df=df[df[self.col_names["col_id_reviewer"]].isin(clientes)]

            clientes_a_eliminar=aux2[self.col_names["col_id_reviewer"]]

            if info:
                t_u = len(df[self.col_names["col_id_reviewer"]].unique())
                t_p = len(df[self.col_names["col_id_product"]].unique())
                print(f"Interaction {iteracion}: \n\tInfo after deleting products bought by less than {self.min[1]} people and ...\n\t... users with less than {self.min[0]} reviews\n\t\tTotal of users: {t_u} \n\t\tTotal of users: {t_p} \n\t\tTotal of reviews: {df.shape[0]} ")

            iteracion+=1

        df[self.col_names["col_id_reviewer"]] = pd.Categorical(df[self.col_names["col_id_reviewer"]]).codes
        df[self.col_names["col_id_product"]] = pd.Categorical(df[self.col_names["col_id_product"]]).codes
        df.to_csv(self.csv_filename, index=False)
        return df

    def create_df(self):
        filename = wget.download(self.url_dataset)
        df = pd.read_csv(filename, delimiter=",", names=[*self.col_names.values()][:4])
        os.remove(filename)
        self.treat_dataset_src(df)


    def getDims(self, df, cols, dataset, col_names):

        if dataset == "movie lens":
            data=df[[col_names["col_id_reviewer"], col_names["col_id_product"], col_names["col_timestamp"]]].astype('int32').to_numpy()
       
        else: 
            data = df[[*cols.values()][:4]].astype('int32').to_numpy()

        add_dims=0

        if dataset != "movie lens":
            for i in range(data.shape[1] - 2):
                # MAKE IT START BY 0
                data[:, i] -= np.min(data[:, i])
                # RE-INDEX
                data[:, i] += add_dims
                add_dims = np.max(data[:, i]) + 1
        else:
            for i in range(data.shape[1] - 1):
                # MAKE IT START BY 0
                data[:, i] -= np.min(data[:, i])
                # RE-INDEX
                data[:, i] += add_dims
                add_dims = np.max(data[:, i]) + 1

        dims = np.max(data, axis=0) + 1
        return data, dims