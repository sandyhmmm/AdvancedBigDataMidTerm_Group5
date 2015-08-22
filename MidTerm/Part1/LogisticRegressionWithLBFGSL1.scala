import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object LogisticRegressionWithLBFGSL1 {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("LBFGSL1")
    val sc = new SparkContext(conf)

// Declaring demlitter
val Delimeter = ","
val textFile = sc.textFile("/user/midterm/YearPredictionMSDClassification.txt")
val data = textFile.map { line =>
val parts = line.split(Delimeter)
	LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble).toArray))		
}

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)


// Declaring number of iterations
val numIterations = 100

//Normalizing data
val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2))

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))

val linearRegSGD = new LogisticRegressionWithLBFGS();
linearRegSGD.optimizer.setNumIterations(numIterations).
  setRegParam(0.01).
  setUpdater(new L1Updater)

 // 
val model = linearRegSGD.run(training2)

val predictionAndLabels = test2.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//calculating accuracy
val accurtimes = predictionAndLabels.filter(r => r._1 == r._2)

val accuracy = accurtimes.count().toFloat/predictionAndLabels.count()

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision


val recall = metrics.recall

println(metrics.confusionMatrix)
println("Recall(0):" + metrics.recall(0.0))

println("Recall(1):" + metrics.recall(1.0))
println("Recall:" + metrics.recall)
println("Precision(0):" + metrics.precision(0.0))
println("Precision(1):" + metrics.precision(1.0))
println("accuracy:" + accuracy)
}
}
