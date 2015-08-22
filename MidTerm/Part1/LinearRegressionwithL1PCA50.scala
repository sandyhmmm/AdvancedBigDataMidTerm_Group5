import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.PCA

object LinearRegressionWithSGDL1PCA {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("LinearRegressionWithSGDL1PCA")
    val sc = new SparkContext(conf)

// Declaring demlitter
val Delimeter = ","
val textFile = sc.textFile("/user/midterm/YearPredictionMSD.txt")
val data = textFile.map { line =>
val parts = line.split(Delimeter)
LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble).toArray))
}



// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val test = splits(1)


// Run training algorithm to build the model
val numIterations = 100

//Normalizing the data
val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2)).cache()

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))

val pca = new PCA(50).fit(training2.map(_.features))

val trainingProjected = training2.map(p => p.copy(features = pca.transform(p.features)))
val testProjected = test2.map(p => p.copy(features = pca.transform(p.features)))

// training the model
val linearRegSGD = new LinearRegressionWithSGD();
linearRegSGD.optimizer.setNumIterations(numIterations).
  setRegParam(0.01).
  setUpdater(new L1Updater).
  setStepSize(0.00001)

val model = linearRegSGD.run(trainingProjected)

val valuesAndPreds = testProjected.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val metrics = new RegressionMetrics(valuesAndPreds)

val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
val meanPred = valuesAndPreds.map{case(v, p) => p}.mean()
val numerator = valuesAndPreds.map{case(v, p) => math.pow((meanPred - p), 2)}.sum()
val varian = numerator.toFloat/valuesAndPreds.count()
println("training Mean Squared Error = " + MSE)
println("Variance:" + varian)
println("RME: "+ metrics.rootMeanSquaredError)
println("Explained Variance: "+metrics.explainedVariance)
}
}
