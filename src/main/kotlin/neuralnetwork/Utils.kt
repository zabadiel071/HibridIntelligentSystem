package neuralnetwork

import kotlin.collections.ArrayList

fun product(inputs: FloatArray?, weights: Array<FloatArray>?): ArrayList<Float>? {
    if (weights != null && inputs != null) {
        val resultSet : ArrayList<Float> = ArrayList()

        val al = weights.toList()
        val bl = inputs.toList()

        al.forEach { t: FloatArray ->
            resultSet.add(t.toList().zip(bl) { a,b -> a*b }.sum())
        }
        return resultSet
    }
    return null
}

fun layerStep(inputs:FloatArray,
              weights: Array<FloatArray>?,
              biases: FloatArray?)
        : List<Float>
{
    val product:ArrayList<Float> = product(inputs, weights)!!

    return product.zip(biases!!.toList()){ a,b -> sigmoid(a - b)}
}

fun sigmoid(x:Float) : Float{
    return (1 / (1 + Math.exp( -x.toDouble() )) ) .toFloat()
}
