import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object KMeansCluster {
  def main(args: Array[String]) {

  	// creating the spark configuration
    val conf = new SparkConf()
    	.setAppName("KMeansCluster")

    // Setting up the configuration to the SparkContext
    val sc = new SparkContext(conf)

    // Loading the libsvm file
	val data = MLUtils.loadLibSVMFile(sc,"data.txt")

	// collecting the features from the RDD
	val parsedData = data.map(s => s.features)

	// Visualizing the data
	val summary: MultivariateStatisticalSummary = Statistics.colStats(parsedData)

	println("Mean")
	println(summary.mean) // a dense vector containing the mean value for each column
	println("variance")
	println(summary.variance) // column-wise variance
	println("non-zeros")
	println(summary.numNonzeros) // number of nonzeros in each column

	//Scaling the data
	val ss = new StandardScaler().fit(parsedData.map(x=>(x)))
	val stdData = parsedData.map(x=>(ss.transform(x)))

	//splitting the data
	val splits = stdData.randomSplit(Array(0.6, 0.4), seed = 11L)
	val training = splits(0)
	val test = splits(1)

	// declaring the normalizer function
	val normalizer = new Normalizer()

	// declaring the static inputs
	val numClusters = 5
	val numIterations = 30
	val initializationMode="random"

	// creating the model
	val clusters = KMeans.train(normalizer.transform(training), numClusters, numIterations)

	// calculating metrics for the mode
	val wsse = clusters.computeCost(normalizer.transform(test))

	//printing the metrics
	println("WSSE:" + wsse)
	println("Cluster centers:" + clusters.clusterCenters)

	}
}
