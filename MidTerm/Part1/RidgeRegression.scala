import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.RidgeRegressionModel
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.mllib.optimization.L2Updater
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object RidgeRegression {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("RidgeRegression")
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
val training = splits(0).cache()
val test = splits(1)


// Run training algorithm to build the model
val numIterations = 100

//Normalizing the data
val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2))

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))

// training the model

val rdg = new RidgeRegressionWithSGD()
rdg.optimizer.setNumIterations(numIterations).
  setRegParam(0.01).
  setStepSize(0.0001)

val model = rdg.run(training2)
val valuesAndPreds = test2.map { point =>
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
