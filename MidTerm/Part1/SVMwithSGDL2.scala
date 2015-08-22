import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object SVMWithSGDL2 {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SVMWithSGDL2")
    val sc = new SparkContext(conf)

//Declaring the delimitter for the file
val Delimeter = ","

//Load the csv file in a RDD
val textFile = sc.textFile("/user/midterm/YearPredictionMSDClassification.txt")
val data = textFile.map { line =>
val parts = line.split(Delimeter)
LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble).toArray))
}

// Split data into training (60%) and test (40%).

val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Setting number of iterations
val numIterations = 100

// Creating new instance for the algorithm
val svmAlg = new SVMWithSGD()

// Setting the parameters for optimizations
svmAlg.optimizer.
  setNumIterations(numIterations).
  setRegParam(0.1)

// normalizing the data
val ss = new StandardScaler().fit(training.map(x=>x.features))

val training1 = training.map(x=>(x.label, ss.transform(x.features)))
val training2 = training1.map(y=> LabeledPoint(y._1,y._2))

val test1 = test.map(x=>(x.label, ss.transform(x.features)))
val test2 = test1.map(y=> LabeledPoint(y._1,y._2))

// Training the model with training data
val model = svmAlg.run(training2)

// Clear the default threshold.
model.clearThreshold()

// Calculating a raw score for the test data
val scoreAndLabels = test2.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

// Get evaluation metrics using BinaryClassificationMetrics cLass
val metrics = new BinaryClassificationMetrics(scoreAndLabels)

// Stroing auROC for performance evaluation
val auROC = metrics.areaUnderROC()

val accurtimes = scoreAndLabels.filter(r => r._1 == r._2)

val accuracy = accurtimes.count()/scoreAndLabels.count()

println("accuracy:" + accuracy)
println("Area under ROC = " + auROC)
}
}
