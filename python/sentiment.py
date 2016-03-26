from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import HashingTF, Tokenizer,RegexTokenizer
from pyspark.sql import Row

arch1 = sc.textFile("../eliana_2016.csv").map(lambda x:x.lower().split("&")[0])
text1 = arch1.map(lambda x: Row(text = x))
teibol = sqlContext.createDataFrame(text1)
teibol.registerTempTable("tuits")
training = sqlContext.sql("select distinct double(1) as label,lower(text) as minus from tuits where lower(text) like '%me encanta%' or lower(text) like '%excelente%' or lower(text) like '%maravill%' or lower(text) like '% buen%' or lower(text) like '%que bien%' or text like '%:)%' or lower(text) like '%:d' or lower(text) like '%:d %' or text like '%;)%' union select distinct double(0) as label,lower(text) as minus from tuits where lower(text) like '%hijue%' or lower(text) like '%malparid%' or lower(text) like '%pésim%' or lower(text) like '%puta%' or lower(text) like '%mierda%' or lower(text) like '%pesim%' union select distinct double(2) as label,lower(text) as minus from tuits where lower(text) like '%capuch%' or lower(text) like '%secuestr%' or lower(text) like '%terror%'")
training.groupby("label").count().show()
test = sqlContext.sql("select *,lower(text) minus from tuits where text is not null")
tokenizer = RegexTokenizer(inputCol="minus", outputCol="words", pattern="\\W+",minTokenLength=3)
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, hashingTF, nb])
model = pipeline.fit(training)
prediction = model.transform(test)
prediction.groupby("prediction").count().show()

