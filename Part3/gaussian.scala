import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object gaussianalgo {
  def main(args: Array[String]) {

    val conf = new SparkConf()
    	.setAppName("Gaussian")
    	.set("spark.executor.memory", "1g")
    val sc = new SparkContext(conf)

	val data = MLUtils.loadLibSVMFile(sc,"spark/midterm/data.txt")
	val parsedData = data.map(s => s.features)
	// Cluster the data into two classes using KMeans
	val ss = new StandardScaler().fit(parsedData.map(x=>(x)))
	val stdData = parsedData.map(x=>(ss.transform(x)))

	val numClusters = 10
	val numIterations = 1
	val initializationMode="random"
	//val clusters = KMeans.train(stdData, numClusters, numIterations)
	val gmm = new GaussianMixture().setK(2).run(stdData)

	for (i <- 0 until gmm.k) {
	  println("weight=%f\nmu=%s\nsigma=\n%s\n" format
	    (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
	}
	}
}
