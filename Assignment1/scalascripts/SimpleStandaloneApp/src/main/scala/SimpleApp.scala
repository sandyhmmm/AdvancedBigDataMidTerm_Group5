/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object SimpleApp {
  def main(args: Array[String]) {
			
			val Delimeter = ",";
			val conf = new SparkConf().setAppName("SVMWithSGD");
    		val sc = new SparkContext(conf);
			val fileLocation = "/Users/Muddassar/Downloads/spark/Assignment1/winequality-white.csv";
			val textFile = sc.textFile(fileLocation).filter(!_.contains("fixed acidity"));
			val data = textFile.map { line =>
			val parts = line.split(Delimeter)
			LabeledPoint(parts(0).toDouble, Vectors.dense(parts.slice(1,5).map(x => x.toDouble).toArray))
			}

			//val Delimeter = ","
			//val data = sc.textFile("park/Assignment1/data.csv").map(line => line.split(Delimeter))

			// Load training data in LIBSVM format.
			//val data = MLUtils.loadLibSVMFile(sc, "spark/Assignment1/data.csv")

			// Split data into training (60%) and test (40%).
			val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
			val training = splits(0).cache()
			val test = splits(1)

			// Run training algorithm to build the model
			val numIterations = 100
			val svmAlg = new SVMWithSGD()
			svmAlg.optimizer.
			  setNumIterations(numIterations).
			  setRegParam(0.1).
			  setUpdater(new L1Updater)
			val model = svmAlg.run(training)
			//val model = SVMWithSGD.train(training, numIterations)

			// Clear the default threshold.
			model.clearThreshold()

			// Compute raw scores on the test set.
			val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
			  val prediction = model.predict(features)
			  (prediction, label)
			}

			val scoreAndLabels = test.map { point =>
			  val score = model.predict(point.features)
			  (score, point.label)
			}

			// Get evaluation metrics.
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			val auROC = metrics.areaUnderROC()

			println("Area under ROC = " + auROC)

			val accurtimes = predictionAndLabels.filter(r => r._1 == r._2)

			val accuracy = accurtimes.count()/predictionAndLabels.count()

			// Get evaluation metrics.
			val metrics1 = new MulticlassMetrics(scoreAndLabels)
			val precision = metrics1.precision


			val recall = metrics1.recall

			println(metrics1.confusionMatrix)
			println("Recall:" + recall)
			println("Precision:" + precision)
			println("accuracy:" + accuracy)
			println("Area under ROC = " + auROC)
  }
}