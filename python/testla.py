from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier,LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer,RegexTokenizer,StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


data = sqlContext.read.json("../data/*.gz")
data.registerTempTable("tuits") 
training = sqlContext.sql("select distinct double(1) as label,lower(text) as text from tuits where lower(text) like '%me encanta%' or lower(text) like '%excelente%' or lower(text) like '%maravill%' or lower(text) like '% buen%' or lower(text) like '%que bien%' or text like '%:)%' or lower(text) like '%:d' or lower(text) like '%:d %' or text like '%;)%' union select distinct double(2) as label,lower(text) as text from tuits where lower(text) like '%hijue%' or lower(text) like '%malparid%' or lower(text) like '%pésim%' or lower(text) like '%puta%' or lower(text) like '%mierda%' or lower(text) like '%pesim%' or lower(text) like '%verga%'")
#training.groupby("label").count().show()
test = sqlContext.sql("select text from tuits where '_corrupt_record' is not null and text is not null")
#tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+",minTokenLength=3)
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
#training.persist()
pipeline = Pipeline(stages=[tokenizer, hashingTF,lr])
model = pipeline.fit(training)
predictions = model.transform(test)
predictions.select("prediction").distinct().show()
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print ("Test Error = %g" % (1.0 - accuracy))

treeModel = model.stages[2]
print (treeModel) # summary only
