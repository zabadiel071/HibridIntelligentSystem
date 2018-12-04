import geneticalgorithm.GeneticAlgorithm
import imageprocessor.MnistReader

fun main(args: Array<String>) {
    println("Desarrollos Inteligentes")

    val imagePosition = 0
    println(
            "Input matrix: ${MnistReader.getOneImage(imagePosition)}"
    ) // get the very first input matrix in position 0 to training in binary format with size 784
    println(
            "Real output of above's matrix: ${MnistReader.getOneLabel(imagePosition)}"
    ) // get the real output in position 0 from the above's matrix

    //Create dummy parents as example for genetic algorithms evolution
    val father = arrayListOf(
            arrayListOf(1.0f,2.0f,3.0f),
            arrayListOf(1.0f,2.0f,3.0f),
            arrayListOf(1.0f,2.0f,3.0f),
            arrayListOf(10.0f,11.0f,12.0f,13.0f)
    )

    val mother = arrayListOf(
            arrayListOf(-1.0f,-2.0f,-3.0f),
            arrayListOf(-1.0f,-2.0f,-3.0f),
            arrayListOf(-1.0f,-2.0f,-3.0f),
            arrayListOf(-10.0f,-11.0f,-12.0f,-13.0f)
    )

    val newChild = GeneticAlgorithm.evolution(father,mother)
    println("New child: $newChild") // crossover and mutation with < 0.005 probability applied
}