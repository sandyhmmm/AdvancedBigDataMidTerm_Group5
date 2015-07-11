import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.PCA

object LogisticRegressionWithSGDAlgo {
def main(args: Array[String]) {
val conf = new SparkConf().setAppName("LogisticRegressionWithSGD")
val sc = new SparkContext(conf)
// Setting number of iterations
val numIterations = 100

val Delimiter = ","
val textFile = sc.textFile("adult_data.csv")
val data = textFile.map { line =>
val parts = line.split(Delimiter)
LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble).toArray))
}

val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2))

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))

//PCA for top 25
//val pca = new PCA(5).fit(training2.map(_.features))

//val trainingProjected = training2.map(p => p.copy(features = pca.transform(p.features)))
//val testProjected = test2.map(p => p.copy(features = pca.transform(p.features)))

// training the model
val model = LogisticRegressionWithSGD.train(training2,numIterations)

val predictionAndLabels = test2.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val accurtimes = predictionAndLabels.filter(r => r._1 == r._2)

val accuracy = accurtimes.count().toFloat/predictionAndLabels.count()

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision


val recall = metrics.recall

println(metrics.confusionMatrix)
println("Recall:" + recall)
println("Precision:" + precision)
println("accuracy:" + accuracy)

}
}
