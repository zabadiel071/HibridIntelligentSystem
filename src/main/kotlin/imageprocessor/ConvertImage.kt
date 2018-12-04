import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

object ConvertImage {
    private lateinit var image: BufferedImage
    var width: Int = 0
    var height: Int = 0

    @JvmStatic
    fun getMatrix(){
        try {
            image = ImageIO.read(javaClass.getResource("blackandwhite.jpg"))
            width = image.width
            height = image.height

            var count = 0
            for (i in 0 until height) {
                for (j in 0 until width) {
                    count++
                    val c = Color(image.getRGB(j, i))
                    println("S.No: " + count + " Red: " + c.red + "  Green: " + c.green + " Blue: " + c.blue)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}