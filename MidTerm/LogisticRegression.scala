import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.mllib.feature.PCA

  case class LabeledDocument(id: Long, text: String, label: Double)
  case class Document(id: Long, text: String)

object LogisticRegression {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("LogisticRegression")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
	

// Declaring demlitter
val Delimeter = ","
val textFile = sc.textFile("/MidTerm/Conversion1.csv")
val textFileZipped = textFile.map(line=>line.split(',')).zipWithIndex()
val data = textFileZipped.map { parts =>
  LabeledDocument(parts._2,parts._1.tail.mkString(" "),parts._1(0).toDouble)  
}

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = data
val test = splits(1)


// Run training algorithm to build the model


//Normalizing the data
/*val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2))

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))*/

// training the model

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")
val hashingTF = new HashingTF()
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("features")
val lr = new LogisticRegression()
  .setMaxIter(1)
  .setRegParam(0.01)
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF, lr))


// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
val crossval = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)

val paramGrid = new ParamGridBuilder()
  .addGrid(hashingTF.numFeatures, Array(25,50))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()
crossval.setEstimatorParamMaps(paramGrid)
crossval.setNumFolds(2)

// Run cross-validation, and choose the best set of parameters.
val cvModel = crossval.fit(training.toDF)

val predictions = cvModel.transform(training.toDF()).select("prediction", "label")

val predictionsAndLabels = predictions.map {case Row(p: Double, l: Double) => (p, l)}

val trainErr = predictionsAndLabels.filter(r => r._1 != r._2).count.toDouble / training.count

println ("trainErr:"+trainErr.toString())
println("Done")

}
}
