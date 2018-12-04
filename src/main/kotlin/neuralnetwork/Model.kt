package neuralnetwork

import java.util.*
import kotlin.collections.HashMap

fun main(args: Array<String>) {
    val nn :NeuralNetwork = NeuralNetwork( listOf(
            TrainPair(floatArrayOf(0.0f,1.0f,1.0f ),floatArrayOf(1.0f, 0.0f)),
            TrainPair(floatArrayOf(0.0f,0.0f,1.0f ),floatArrayOf(1.0f, 0.0f)),
            TrainPair(floatArrayOf(1.0f,1.0f,0.0f ),floatArrayOf(0.0f, 1.0f)),
            TrainPair(floatArrayOf(1.0f,0.0f,1.0f ),floatArrayOf(0.0f, 1.0f))
        )
    )
    nn.train()

    print(nn.weights)
}


data class TrainPair(val input: FloatArray, val output: FloatArray)


class NeuralNetwork(val trainSet : List<TrainPair>){
    var weights: HashMap<String, Array<FloatArray>>
    var biases : HashMap<String, FloatArray >

    init {
        weights = setWeights()
        biases = setBiases()
    }

    /**
     *
     */
    fun train(){
        var layer1:FloatArray
        var layer2:FloatArray
        var layer3:FloatArray
        var output:FloatArray

        var corrections: HashMap<String, Array<FloatArray>> = HashMap()

        var errors:FloatArray
        var gradients:FloatArray
        var _correction:Array<FloatArray>

        for (i in 0..neurons.iterations){
            trainSet.forEach { trainPair: TrainPair ->
                layer1 = layerStep(trainPair.input, weights["w1"],biases["b1"] ).toFloatArray()
                layer2 = layerStep(layer1, weights["w2"],biases["b2"] ).toFloatArray()
                layer3 = layerStep(layer2, weights["w3"], biases["b3"] ).toFloatArray()
                output = layerStep(layer3 , weights["out"], biases["out"]).toFloatArray()

                errors = getErrors(trainPair.output, output)
                gradients = getGradients(output, errors)

                _correction = gradientCorrection(gradients, weights["out"]!!)
                corrections["out"] =getCorrection(_correction, layer3)

                gradients = getGradients(layer3, subGradient(weights["out"]!!, gradients))
                _correction = gradientCorrection(gradients, weights["w3"]!!)
                corrections["w3"] =getCorrection(_correction, layer2)

                gradients = getGradients(layer2, subGradient(weights["w3"]!!, gradients))
                _correction = gradientCorrection(gradients, weights["w2"]!!)
                corrections["w2"] = getCorrection(_correction, layer1)

                gradients = getGradients(layer1, subGradient(weights["w2"]!!, gradients))
                _correction = gradientCorrection(gradients , weights["w1"]!!)
                corrections["w1"] = getCorrection(_correction, trainPair.input)


                updateWeights(corrections)
            }
        }
    }

    private fun updateWeights(corrections: HashMap<String, Array<FloatArray>>) {
        weights["out"] = weights["out"]!!.zip(corrections["out"]!!){w: FloatArray, delta: FloatArray ->
            w.zip(delta){a: Float, b: Float -> a + b }.toFloatArray()
        }.toTypedArray()
        weights["w3"] = weights["w3"]!!.zip(corrections["w3"]!!){w: FloatArray, delta: FloatArray ->
            w.zip(delta){a: Float, b: Float -> a + b }.toFloatArray()
        }.toTypedArray()
        weights["w2"] = weights["w2"]!!.zip(corrections["out"]!!){w: FloatArray, delta: FloatArray ->
            w.zip(delta){a: Float, b: Float -> a + b }.toFloatArray()
        }.toTypedArray()
        weights["w1"] = weights["w1"]!!.zip(corrections["out"]!!){w: FloatArray, delta: FloatArray ->
            w.zip(delta){a: Float, b: Float -> a + b }.toFloatArray()
        }.toTypedArray()
    }

    /**
     *
     */
    fun getCorrection(_correction: Array<FloatArray>, layer: FloatArray): Array<FloatArray> {
        return _correction.map { floats ->
            floats.zip(layer){a, b -> a*b*neurons.learningRate }.toFloatArray()
        } .toTypedArray()
    }

    fun gradientCorrection(gradients: FloatArray, weights: Array<FloatArray>): Array<FloatArray> {
        return gradients.zip(weights!!){
            g, w -> w.map { fl -> fl*g }.toFloatArray()
        }.toTypedArray()
    }

    fun subGradient(weights: Array<FloatArray>, gradients: FloatArray): FloatArray {
        return  traspose(weights!!)
                .map { floats ->
                    gradients.zip(floats){ a, b -> a*b }.sum()
                }
                .toFloatArray()
    }

    fun traspose(arr: Array<FloatArray>) : Array<FloatArray> {
        val traspose = Array(arr[0].size){FloatArray(arr.size)}

        for(i in 0 until  arr.size){
            for (j in 0 until arr[0].size){
                traspose[j][i] = arr[i][j]
            }
        }

        return traspose
    }


    /**
     * Gradients for output layer
     */
    fun getGradients(output: FloatArray, errors:FloatArray) = output.zip(errors){ a,b -> a*(1-a)*b}.toFloatArray()

    /**
     *
     */
    fun getErrors(desiredOutputs: FloatArray, realOutputs: FloatArray) =  desiredOutputs.zip(realOutputs){ a, b-> a-b}.toFloatArray()

    fun setWeights() : HashMap<String, Array<FloatArray>> {
        val w1 = Array(neurons.hidden1, {FloatArray(neurons.inputs,{Random().nextFloat() * 0.001f})})
        val w2 = Array(neurons.hidden2, {FloatArray(neurons.hidden1,{Random().nextFloat() * 0.001f})})
        val w3 = Array(neurons.hidden3, {FloatArray(neurons.hidden2,{Random().nextFloat() * 0.001f})})
        //val out = Array(neurons.outputs, {FloatArray(neurons.hidden3,{Random().nextFloat() * 0.001f})})
        val out = Array(neurons.outputs, {FloatArray(neurons.hidden2,{Random().nextFloat() * 0.001f})})

        return hashMapOf(
                Pair("w1", w1),
                Pair("w2", w2),
                Pair("w3", w3),
                Pair("out", out)
        )
    }

    fun setBiases() : HashMap<String, FloatArray> {
        val b1 = FloatArray(neurons.hidden1, { _ -> Random().nextFloat()})
        val b2 = FloatArray(neurons.hidden2, { _ -> Random().nextFloat()})
        val b3 = FloatArray(neurons.hidden3, { _ -> Random().nextFloat()})
        val out = FloatArray(neurons.outputs, { _ -> Random().nextFloat()})

        return hashMapOf(
                Pair("b1", b1 ),
                Pair("b2", b2 ),
                Pair("b3", b3 ),
                Pair("out", out )
        )
    }
}