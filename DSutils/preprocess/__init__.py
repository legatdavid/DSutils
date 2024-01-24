# Databricks notebook source
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
    model = pipeline.fit(train_sample)
    scored = model.transform(train_sample)

    importances=model.stages[-1].featureImportances
    list_extract = []
    for i in scored.schema['features'].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + scored.schema['features'].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: importances[x])
    varlist['feature'] = varlist['name'].apply(lambda x: x[:(x+'_enc').index('_enc')])
    FeatureImportance = varlist.groupby('feature', as_index=False).sum('score').sort_values('score', ascending = False)
    FeatureImportance = FeatureImportance.assign(row_number=range(len(FeatureImportance)))
    numerical_features = list(FeatureImportance[FeatureImportance['feature'].isin(continuous_cols) & (FeatureImportance['row_number'] < numfeaatures)]['feature'])
    
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
    model = pipeline.fit(train_sample)
    scored = model.transform(train_sample)

    importances=model.stages[-1].featureImportances
    list_extract = []
    for i in scored.schema['features'].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + scored.schema['features'].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: importances[x])
    varlist['feature'] = varlist['name'].apply(lambda x: x[:(x+'_enc').index('_enc')])
    FeatureImportance = varlist.groupby('feature', as_index=False).sum('score').sort_values('score', ascending = False)
    FeatureImportance = FeatureImportance.assign(row_number=range(len(FeatureImportance)))
    categorical_features = list(FeatureImportance[FeatureImportance['feature'].isin(categorical_cols) & (FeatureImportance['row_number'] < features_RF)]['feature']) 
    numerical_features = list(FeatureImportance[FeatureImportance['feature'].isin(continuous_cols) & (FeatureImportance['row_number'] < features_RF)]['feature'])

    return categorical_features, numerical_features
