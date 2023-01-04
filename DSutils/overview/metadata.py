# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf


def Metadata(data=None, interval_thrs=20, missing_thrs=0.5, string_null=["XNA", ""], ):
    
    if data == None:
        raise BaseException("Input attribute 'data' of function 'Meta' is mandatory.")
    if type(string_null) == "string":
        string_null=[string_null]
    spark = SparkSession.getActiveSession()
    rec = data.count()
    meta = data.dtypes
    Metadata = spark.createDataFrame(meta, schema=["Column_Name", "Column_Type"])
    print(len([c for c in meta]))
    valno = spark.createDataFrame(data.groupby()
                                       .agg(*(sf.approxCountDistinct(sf.col(c[0])).alias(c[0]) for c in meta))
                                       .toPandas().T.reset_index(), 
                                  schema=["Column_Name","Value_Count"]
                                 )

    missno = spark.createDataFrame(data.groupby()
                                        .agg(*(sf.sum(sf.when(sf.col(c[0]).isNull(),1).otherwise(0)).alias(c[0]) for c in meta))
                                        .toPandas().T.reset_index(), 
                                   schema=["Column_Name","Missing_Count"]
                                  )

#     if len([c for c in meta if (c[1]=='string')]) > 0:
#         missno = missno.join(
#                  spark.createDataFrame(data.groupby()
#                                             .agg(*(sf.sum(sf.when(sf.col(c[0]).isin(string_null), 1).otherwise(0)).alias(c[0]) for c in meta if (c[1] == 'string')))
#                                             .toPandas().T.reset_index(), 
#                                        schema=["Column_Name","Missing_Count_str"]
#                                       ),
#                     on="Column_Name", how="left").withColumn("Missing_Count", sf.col("Missing_Count") + sf.when(sf.col("Missing_Count_str").isNull(),0).otherwise(sf.col("Missing_Count_str"))
#                 ).drop("Missing_Count_str")

    Metadata = Metadata.join(valno, on="Column_Name").join(missno, on="Column_Name")
    Metadata = Metadata.withColumn("Level", 
                              sf.when(sf.col("Column_Type") == "string" ,"Nominal")
                                .when((sf.col("Column_Type") != "string") & (sf.col("Value_Count") <= interval_thrs), "Nominal")
                                .otherwise("Interval"))
    Metadata = Metadata.withColumn("Role", 
                              sf.when(sf.col("Missing_Count") / rec > missing_thrs, "Rejected")
                                .when((sf.col("Value_Count") == 1), "Rejected")
                                .otherwise("Feature"))
    return Metadata

#     def __init__(self, data):
#         print("TBD")

#     def set_target(self, columns):
#         print("TBD")

#     def set_rejected(self, columns):
#         print("TBD")

#     def set_category(self, columns):
#         print("TBD")

