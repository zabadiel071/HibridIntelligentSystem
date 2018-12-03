package neuralnetwork

import java.util.*

fun main(args: Array<String>) {
    val nn = NeuralNetwork( listOf(
            TrainPair(floatArrayOf(0.0f,1.0f,1.0f ),floatArrayOf(0.0f)),
            TrainPair(floatArrayOf(0.0f,0.0f,1.0f ),floatArrayOf(0.0f)),
            TrainPair(floatArrayOf(1.0f,1.0f,0.0f ),floatArrayOf(1.0f)),
            TrainPair(floatArrayOf(1.0f,0.0f,1.0f ),floatArrayOf(1.0f))
        )
    ).train()
}

object neurons{
    val inputs:Int = 3  //784
    val hidden1:Int = 4 //512
    val hidden2:Int = 3 //256
    val hidden3:Int = 2 //128
    val outputs:Int = 1

    val learningRate = 0.01f
    val iterations = 1000
}

data class TrainPair(val input: FloatArray, val output: FloatArray)


class NeuralNetwork(val trainSet : List<TrainPair>){
    var weights: HashMap<String, Array<FloatArray>>
    var biases : HashMap<String, FloatArray >

    init {
        weights = weights()
        biases = biases()
    }

    fun train(){
        trainSet.forEach { trainPair: TrainPair ->

            val layer1 = layerStep(trainPair.input, weights["w1"],biases["b1"] )
            val layer2 = layerStep(layer1.toFloatArray(), weights["w2"],biases["b2"] )
            val layer3 = layerStep(layer2.toFloatArray(), weights["w3"],biases["b3"] )

            val output = layerStep(layer3.toFloatArray(), weights["out"], biases["out"])

            val errors = trainPair.output.zip(output){a,b-> a-b}

            val gradients = output.zip(errors){ a,b -> a*(1-a)*b}

            // TODO: Update weights (Backpropagation)
            weights["out"]!!.forEach {  floats ->

                floats.map { wi -> wi }
            }

            print(gradients)
        }
    }

    fun weights() : HashMap<String, Array<FloatArray>> {
        val w1 = Array(neurons.hidden1, {FloatArray(neurons.inputs,{Random().nextFloat() * 0.001f})})
        val w2 = Array(neurons.hidden2, {FloatArray(neurons.hidden1,{Random().nextFloat() * 0.001f})})
        val w3 = Array(neurons.hidden3, {FloatArray(neurons.hidden2,{Random().nextFloat() * 0.001f})})
        val out = Array(neurons.outputs, {FloatArray(neurons.hidden3,{Random().nextFloat() * 0.001f})})

        return hashMapOf(
                Pair("w1", w1),
                Pair("w2", w2),
                Pair("w3", w3),
                Pair("out", out)
        )
    }

    fun biases() : HashMap<String, FloatArray> {
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