package geneticalgorithm

import java.text.DecimalFormat
import java.util.Random
import kotlin.collections.ArrayList

object GeneticAlgorithm{

    @JvmStatic
    fun evolution(
            father: ArrayList<ArrayList<Float>>,
            mother: ArrayList<ArrayList<Float>>
    ): ArrayList<ArrayList<Float>>{
        val offspring = crossover(father,mother)
        val mutatedOffspring = mutation(offspring)
        return mutatedOffspring
    }

    private fun crossover(
            father: ArrayList<ArrayList<Float>>,
            mother: ArrayList<ArrayList<Float>>
    ): ArrayList<ArrayList<Float>> {
        val offspring: ArrayList<ArrayList<Float>> = ArrayList()
        for (i in 0 until father.size){
            val innerList: ArrayList<Float> = ArrayList()
            offspring.add(innerList)
            if (Math.random() < 0.5)
                for (j in 0 until father[i].size)
                    offspring[i].add(father[i][j])
            else
                for (j in 0 until mother[i].size)
                    offspring[i].add(mother[i][j])
        }
        return offspring
    }

    private fun mutation(offspring: ArrayList<ArrayList<Float>>): ArrayList<ArrayList<Float>>{
        if (Math.random() < 0.005){
            val geneToModify = Random().nextInt(offspring.size - 0) + 0 // random int between min,max genes
            println("The gene $geneToModify has mutated!")
            for (i in 0 until offspring[geneToModify].size){
                val modification = DecimalFormat("#.00").format(
                        ((Math.random() * (1 - (-1) )) - 1)
                ).toFloat() //random double between -1,1 || limit 2 decimals
                offspring[geneToModify][i] += modification
            }
        }
        return offspring
    }
}