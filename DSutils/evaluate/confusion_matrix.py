# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf

# COMMAND ----------

def grouped_stats(prediction=None, bins=10, labelCol="label", predictCol="prediction", confusion_matrix=True, cum_lift=True, accuracy=False, precision=False, recall=False, sensitivity=False, specidficity=False ):

    prediction = prediction.withColumn("prob1", sf.element_at(vector_to_array(predictCol), 2))
    bin_stats = (prediction.withColumn("rank", sf.ntile(10).over(Window.orderBy(sf.desc("prob1"))))
             .select("prob1", "rank", labelCol)
             .groupBy("rank")
             .agg(sf.count(labelCol).alias("num_subj"),
                  sf.sum(labelCol).alias("num_label"),
                  sf.avg("prob1").alias("avg_probability")
                  )
             .withColumn("cum_sum_subj",
                         sf.sum("num_subj").over(Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0))
                         )
             .withColumn("cum_sum_label",
                         sf.sum("num_label").over(Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0))
                         )
             .withColumn("cum_avg_label",
                         sf.avg("num_label").over(Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0))
                         )
            )
    
    if confusion_matrix is True:
        totals = bin_stats.agg(sf.sum("num_subj").alias("tot_num_subj"),
                               sf.sum("num_label").alias("tot_num_label")
                              ).collect()[0]
        bin_stats = (bin_stats
             .withColumn("TP", sf.col("cum_sum_label") )
             .withColumn("FP", sf.col("cum_sum_subj") - sf.col("cum_sum_label") )
             .withColumn("FN", totals["tot_num_label"] - sf.col("cum_sum_label") )
             .withColumn("TN", totals["tot_num_subj"] - totals["tot_num_label"] - sf.col("cum_sum_subj") + sf.col("cum_sum_label"))
            )
        
    if cum_lift is True:
        avg_lead_rate = bin_stats.filter(sf.col("rank") == 10).select("cum_avg_label").collect()[0][0]
        bin_stats = (bin_stats
             .withColumn("cum_lift", sf.col("cum_avg_label").cast("double") / avg_lead_rate)

