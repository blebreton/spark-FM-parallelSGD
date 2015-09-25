import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._ 
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, Vector => SparkV, SparseVector => SparkSV, DenseVector => SparkDV, Matrix => SparkM }
import breeze.linalg.{Vector => BV, SparseVector => BSV, DenseVector => BDV, DenseMatrix => BDM, _}
import breeze.numerics._
import scala.math._
import util.Random.shuffle


implicit def toBreeze(v: SparkV) : BV[Double]= {
    /** Convert a spark.mllib.linalg vector into breeze.linalg.vector
    * We use Breeze library for any type of matrix operations because mllib local linear algebra package doesn't have any support for it (in Spark 1.4.1)
    * mllib toBreeze function already exists but is private to mllib scope
    *
    * Automatically choose the representation (dense or sparse) which use the less memory to be store
    */
        val nnz = v.numNonzeros
        if (1.5 * (nnz +1.0) < v.size) {
            new BSV(v.toSparse.indices, v.toSparse.values, v.size)
        } else {
            BV(v.toArray)
        }
        
}

def fm_get_p (X: SparkV, W : BDM[Double]) : Double = {
    val nnz = X.numNonzeros
    var x:BV[Double] = BV(0.)
    var w:BDM[Double] = BDM(0.)
    /**	Computes the probability of an instance given a model */
    
    // convert Spark Vector to Breeze Vector
    if (1.5 * (nnz +1.0) < X.size) {
        val xsp = X.toSparse
        val xind = xsp.indices.toSeq
        x = BV(xsp.values)
        w = W(xind,::).toDenseMatrix
    } else {
        x = X:BV[Double]
        w = W
    }
    
	val xa = x.toDenseVector.asDenseMatrix
	val VX =  xa * w
	val VX_square = (xa :* xa)  * (w :* w)
	
	val phi = 0.5*(VX:*VX - VX_square).sum
	return 1/(1 + exp(-phi))

}

def predictFM (data : RDD[LabeledPoint], W : BDM[Double]) : RDD[Double] = {
    /** Computes the probabilities given a model for the complete data set */
    return data.map(row => fm_get_p(row.features, W))
}

def logloss(y_pred : Array[Double], y_true : Array[Int]) : Double = {
    /* Computes the logloss given the true label and the predictions */
    val losses =  BDV(y_true.map(v => v.toDouble) :* (y_pred.map(log))) + BDV(y_true.map(v => (1-v).toDouble) :* (y_pred.map(v => 1-v).map(log)))
	return -losses.sum / losses.size
}

def evaluate (data : RDD[LabeledPoint], w : BDM[Double]) : Double = {
    /*  Evaluate a Factorization Machine model on a data set.
    *
    *    Parameters:
    *    data : RDD of LabeledPoints
    *            Evaluation data. Labels should be -1 and 1
    *    w : Breeze dense matrix
    *            FM model, result from trainFM_sgd or trainFM_parallel_sgd
    *   return:
        logl: average logloss
    */
        val y_true_rdd = data.map(lp => if(lp.label == 1){1} else {0})
        val y_true = y_true_rdd.collect()
        val y_pred_rdd = predictFM(data, w)
        val y_pred = y_pred_rdd.collect()
        
        val logl = logloss(y_pred, y_true)

        return logl
}

def fm_gradient_sgd_trick (X: BV[Double], y: Double , W: BDM[Double], regParam: Double): BDM[Double] = {
	/* 	Computes the gradient for one instance using Rendle FM paper (2010) trick (linear time computation) */
	val nrFeat = X.size
	val xa = X.toDenseVector.asDenseMatrix
	val x_matrix = xa.t * xa
	val VX =  xa * W
	val VX_square = (xa :* xa)  * (W :* W)

	val phi = 0.5*(VX:*VX - VX_square).sum
	val expnyt = exp(-y*phi)
	var i = 0
	while (i < nrFeat) {
		x_matrix.update(i, i, 0.)
		i += 1
	}
	
	var result = x_matrix * W :*(-y*expnyt)/(1+expnyt)

	result+= W:*regParam

	return result
}

def sgd_subset(train_X : Array[SparkV], train_Y : Array[Double], W : BDM[Double], iter_sgd : Int, alpha : Double, regParam : Double) : BDM[Double] =  {
    /*    Computes stochastic gradient descent for a partition (in memory) */
    
    val N = train_X.length
    var wsub : BDM[Double] = BDM.zeros(W.rows,W.cols)
    wsub += W
    var G = BDM.ones[Double](W.rows,W.cols)

    for (i <- 1 to iter_sgd) {
        var random_idx_list = shuffle(0 to N-1)
        for (j <- 0 to N-1) {
            val idx = random_idx_list(j)
            val X = train_X(idx)
            val y = train_Y(idx)
            val nnz = X.numNonzeros
            if (1.5 * (nnz +1.0) < X.size) {
                val xsp = X.toSparse
                val xind = xsp.indices.toSeq
                val grads_compress = fm_gradient_sgd_trick(BV(xsp.values), y, wsub(xind,::).toDenseMatrix, regParam)
                G(xind,::) := (G(xind,::).toDenseMatrix + (grads_compress :* grads_compress))
                wsub(xind,::) := wsub(xind,::).toDenseMatrix - (alpha :* (grads_compress :/ (G(xind,::).toDenseMatrix.map(sqrt(_)))))
                
            } else {
                val grads = fm_gradient_sgd_trick(X, y, wsub, regParam)    
            
                G += grads :* grads
                wsub -= alpha * grads :/ (G.map(sqrt(_)))

            }
        }
    }
    return wsub
}

def trainFM_parallel_sgd (data : RDD[LabeledPoint], iterations: Int = 50, iter_sgd : Int =5, alpha : Double =0.01, regParam : Double = 0., factorLength : Int = 4, verbose: Boolean =false) : BDM[Double] = { 
    /*
    * Train a Factorization Machine model using parallel stochastic gradient descent.
    *
    * Parameters:
    * data : RDD of LabeledPoints
    *    Training data. Labels should be -1 and 1
    *    Features should be Vector from mllib.linalg library
    * iterations : Int
    *    Nr of iterations of parallel SGD. default=50
    * iter_sgd : Int
    *	Nr of iteration of sgd in each partition. default = 5
    * alpha : Double
    *    Learning rate of SGD. default=0.01
    * regParam : Double 
    *    Regularization parameter. default=0.01
    * factorLength : Int
    *    Length of the weight vectors of the FMs. default=4
    * verbose: Boolean
    *    Whether to ouptut iteration numbers, time, logloss for train and validation sets
    * returns: W
    *    Breeze dense matrix holding the model weights
    */
    val train = data
    val valid = data
    if (verbose) {
        val Array(train,valid) = data.randomSplit(Array(0.8, 0.2))
        valid.cache()
    } 
    train.cache()
    
    val train_X = train.map(xy => xy.features).glom()
    val train_Y = train.map(xy => xy.label).glom()
    val train_XY = train_X.zip(train_Y)
    train_XY.cache()
    
    val nrFeat = train_XY.first()._1(0).size
    var W = BDM.rand(nrFeat,factorLength)
    W :*= 1 / sqrt(sum(W:*W))

    if (verbose) {
        println("iter   train_logl  valid_logl")
        println("%d         %.5f   %.5f".format(0, evaluate(train, W), evaluate(valid, W)))

    }
    
    for (i <- 1 to iterations) {
        val wb = sc.broadcast(W)
        val wsub = train_XY.map(xy => sgd_subset(xy._1, xy._2, wb.value, iter_sgd, alpha, regParam))
        W = wsub.map(w=> w.map(_/5)).reduce(_+_)
        if (verbose) {
            println("%d         %.5f   %.5f".format(i, evaluate(train, W), evaluate(valid, W)))
        }
    }
    
    train_XY.unpersist()
    
    return W
}

