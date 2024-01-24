# Databricks notebook source
def BestLogRegClassifier(
    df_train,
    inputCategoryCols: list = None,
    inputNumericCols: list = None,
    labelCol: str = "label",
):
  
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

    pipe = [StringIndexer(inputCol=x , outputCol=x+'_idx', handleInvalid = 'keep') for x in inputCategoryCols]
    pipe.extend([OneHotEncoder(inputCol=x+'_idx', outputCol=x+'_enc') for x in inputCategoryCols])
    pipe.append(VectorAssembler(inputCols=[x+'_enc' for x in inputCategoryCols] + list(inputNumericCols) ,outputCol='features'))
    pipe.append(LogisticRegression(labelCol=labelCol, elasticNetParam=1))
    pipeline = Pipeline(stages=pipe)
    
    parms = ParamGridBuilder() \
        .addGrid(pipeline.getStages()[-1].regParam, [0, 0.01]) \
        .build()
    eval = BinaryClassificationEvaluator(labelCol=labelCol)
    tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=parms, evaluator=eval, trainRatio=0.7)
    model = tvs.fit(df_train).bestModel

    return model

# COMMAND ----------

def BestGBTClassifier(
    df_train,
    inputCategoryCols: list = None,
    inputNumericCols: list = None,
    labelCol: str = "label",
):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

    pipe = [StringIndexer(inputCol=x , outputCol=x+'_idx', handleInvalid = 'keep') for x in inputCategoryCols]
    pipe.extend([OneHotEncoder(inputCol=x+'_idx', outputCol=x+'_enc') for x in inputCategoryCols])
    pipe.append(VectorAssembler(inputCols=[x+'_enc' for x in inputCategoryCols] + list(inputNumericCols) ,outputCol='features'))
    pipe.append(GBTClassifier(labelCol=labelCol, maxBins=8))
    pipeline = Pipeline(stages=pipe)
            
    parms = (ParamGridBuilder()
      .addGrid(pipeline.getStages()[-1].stepSize , [0.1, 0.25, 0.5])
      .build())
    eval = BinaryClassificationEvaluator(labelCol=labelCol)
    tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=parms, evaluator=eval, trainRatio=0.7)
    model = tvs.fit(df_train).bestModel

    return model
