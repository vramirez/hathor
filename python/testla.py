from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import HashingTF, Tokenizer,RegexTokenizer

data_ori = sqlContext.read.json("../data/*gz")
textol=data_all.select("text")
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+",minTokenLength=3)
hashingTF = HashingTF(numFeatures= 1 << 18,inputCol=tokenizer.getOutputCol(), outputCol="features")
dat2= data_ori.select("text").withColumnRenamed("text","tuit")