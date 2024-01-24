# Databricks notebook source
import pyspark.sql.functions as sf

def lift_curve_spark(
    df_predictions,
    bin_count: int = 10,
    label_col: str = "label",
    probability_col: str = None,
):
  from pyspark.sql.window import Window
  from pyspark.ml.functions import vector_to_array
  
  if type(bin_count) != int or bin_count < 0:
      raise ValueError("Invalid 'bin_count' param value.Expected int >= 0, got '{}'.".format(bin_count))

  if not probability_col:
      if "rawPrediction" in df_predictions.columns:
          probability_col = sf.element_at(vector_to_array("probability"), 2)
      elif "prediction" in df_predictions.columns:
          probability_col = sf.col("prediction")
      else:
          raise ValueError("Param 'probability_col' not supplied and could not infer column name.")
  else:
      if probability_col not in df_predictions.columns:
          raise ValueError("Value given in 'probability_col' is not among dataframe columns.")

      _dtype = [_dtype for col, _dtype in df_predictions.dtypes if col == probability_col][0]
      if _dtype == "vector":
          probability_col = sf.element_at(vector_to_array(probability_col), 2)
      else:
          probability_col = sf.col(probability_col)

  df_lift = (df_predictions
             .select(probability_col.alias("class_1_probability"), label_col)
             .withColumn("rank", sf.ntile(bin_count).over(Window.orderBy(sf.desc("class_1_probability"))))
             .select("class_1_probability", "rank", label_col)
             .groupBy("rank")
             .agg(sf.count(label_col).alias("num_subj"),
                  sf.sum(label_col).alias("num_target"),
                  sf.avg("class_1_probability").alias("avg_probability")
                  )
             .withColumn("cum_avg_target",
                         sf.avg("num_target").over(Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0))
                         )
             )

  avg_lead_rate = df_lift.filter(sf.col("rank") == bin_count).select("cum_avg_target").collect()[0][0]

  df_cum_lift = (df_lift
                 .withColumn("cum_lift", sf.col("cum_avg_target").cast("double") / avg_lead_rate)
                 .selectExpr("rank as bucket", "num_subj", "num_target",
                             "avg_probability", "cum_avg_target", "cum_lift"
                             )
                 )

  return df_cum_lift

def ClassifierValidation(
    df_validation,
    model,
    label_col: str = "label",
):
  from pyspark.ml.evaluation import BinaryClassificationEvaluator

  scored = model.transform(df_validation).select(label_col, "probability")

  evaluatorBin = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="probability")

  auROC = evaluatorBin.evaluate(scored, {evaluatorBin.metricName: "areaUnderROC"})
  auPR = evaluatorBin.evaluate(scored, {evaluatorBin.metricName: "areaUnderPR"})
  lift = lift_curve_spark(df_predictions=scored, label_col=label_col, probability_col='probability').filter(sf.col('bucket')==1).select('cum_lift').first()[0]
  
  return auROC, auPR, lift
