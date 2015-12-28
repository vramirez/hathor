from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import HashingTF, Tokenizer,RegexTokenizer,StringIndexer,VectorIndexer

data = sqlContext.read.json("../data/*.gz")
data.persist() 
data.registerTempTable("tuits") 
training = sqlContext.sql("select distinct double(3) as label,lower(text) as text from tuits where lower(text) like '%me encanta%' or lower(text) like '%excelente%' or lower(text) like '%maravill%' or lower(text) like '% buen%' or lower(text) like '%que bien%' or text like '%:)%' or lower(text) like '%:d' or lower(text) like '%:d %' or text like '%;)%' union select distinct double(2) as label,lower(text) as text from tuits where lower(text) like '%hijue%' or lower(text) like '%malparid%' or lower(text) like '%pésim%' or lower(text) like '%puta%' or lower(text) like '%mierda%' or lower(text) like '%pesim%' or lower(text) like '% mal%'")
training.persist()
test = sqlContext.sql("select text from tuits where '_corrupt_record' is not null and text is not null")
test.persist()
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+",minTokenLength=3)
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
dt = DecisionTreeClassifier(maxDepth=5, labelCol="indexed",featuresCol="features")
pipeline = Pipeline(stages=[tokenizer, hashingTF, stringIndexer,dt])
model = pipeline.fit(training)
result= model.transform(test)
result.groupby("prediction").count().show()

