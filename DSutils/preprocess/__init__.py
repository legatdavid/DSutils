# Databricks notebook source
def UnivarNumFeatureSelector(
    df,
    inputCols: list = None,
    labelCol: str = "label",
    numfeaatures: int = 100,
):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler, UnivariateFeatureSelector
    from pyspark.ml import Pipeline
    
    ufs = UnivariateFeatureSelector(featuresCol='features', labelCol=labelCol, outputCol="selectedFeatures")
    ufs.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(numfeaatures)

    pipe = [VectorAssembler(inputCols=inputCols ,outputCol='features'), ufs]
    pipeline = Pipeline(stages=pipe)
    model = pipeline.fit(train_sample)
    return model.stages[-1].selectedFeatures

# COMMAND ----------

def RFNumFeatureSelector(
    df,
    inputCols: list = None,
    labelCol: str = "label",
    numfeaatures: int = 100,
):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    
    pipe = [VectorAssembler(inputCols=inputCols ,outputCol='features'),
            RandomForestClassifier(featuresCol='features', labelCol=labelCol, numTrees=10, maxDepth=10, subsamplingRate=0.7)]
    pipeline = Pipeline(stages=pipe)
    model = pipeline.fit(df)
    scored = model.transform(df)

    importances=model.stages[-1].featureImportances
    list_extract = []
    for i in scored.schema['features'].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + scored.schema['features'].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: importances[x])
    varlist['feature'] = varlist['name'].apply(lambda x: x[:(x+'_enc').index('_enc')])
    FeatureImportance = varlist.groupby('feature', as_index=False).sum('score').sort_values('score', ascending = False)
    FeatureImportance = FeatureImportance.assign(row_number=range(len(FeatureImportance)))
    numerical_features = list(FeatureImportance[FeatureImportance['feature'].isin(inputCols) & (FeatureImportance['row_number'] < numfeaatures)]['feature'])
    
    return numerical_features

# COMMAND ----------

def RFFeatureSelector(
    df,
    inputCategoryCols: list = None,
    inputNumericCols: list = None,
    labelCol: str = "label",
    numfeaatures: int = 100,
):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline

    pipe = [StringIndexer(inputCol=x , outputCol=x+'_idx') for x in inputCategoryCols]
    pipe.extend([OneHotEncoder(inputCol=x+'_idx', outputCol=x+'_enc') for x in inputCategoryCols])
    pipe.append(VectorAssembler(inputCols=[x+'_enc' for x in inputCategoryCols] + list(inputNumericCols) ,outputCol='features'))
    pipe.append(RandomForestClassifier(featuresCol='features', labelCol=labelCol, numTrees=10, maxDepth=10, subsamplingRate=0.7))
    pipeline = Pipeline(stages=pipe)
    model = pipeline.fit(df)
    scored = model.transform(df)

    importances=model.stages[-1].featureImportances
    list_extract = []
    for i in scored.schema['features'].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + scored.schema['features'].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: importances[x])
    varlist['feature'] = varlist['name'].apply(lambda x: x[:(x+'_enc').index('_enc')])
    FeatureImportance = varlist.groupby('feature', as_index=False).sum('score').sort_values('score', ascending = False)
    FeatureImportance = FeatureImportance.assign(row_number=range(len(FeatureImportance)))
    categorical_features = list(FeatureImportance[FeatureImportance['feature'].isin(inputCategoryCols) & (FeatureImportance['row_number'] < numfeaatures)]['feature']) 
    numerical_features = list(FeatureImportance[FeatureImportance['feature'].isin(inputNumericCols) & (FeatureImportance['row_number'] < numfeaatures)]['feature'])

    return categorical_features, numerical_features
