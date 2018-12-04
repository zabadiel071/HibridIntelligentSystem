package imageprocessor

import util.Constants
import java.io.ByteArrayOutputStream
import java.io.RandomAccessFile
import java.lang.String.format
import java.nio.ByteBuffer
import kotlin.experimental.and

object MnistReader {
    val LABEL_FILE_MAGIC_NUMBER = 2049
    val IMAGE_FILE_MAGIC_NUMBER = 2051

    fun getOneImage(id: Int): ArrayList<Int> {
        val images = getImages(Constants.IMAGES_TRAINING_FILE)
        val imageInput: ArrayList<Int> = ArrayList()
        images[id].forEach { x ->
            x?.forEach {
                if(it != 0) imageInput.add(1) else imageInput.add(it)
            }
        }
        return imageInput
    }

    fun getOneLabel(id: Int): Int{
        val labels = getLabels(Constants.LABELS_TRAINING_FILE)
        return labels[id]
    }

    fun getLabels(infile: String): IntArray {

        val bb = loadFileToByteBuffer(infile)

        assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.int)

        val numLabels = bb.int
        val labels = IntArray(numLabels)

        for (i in 0 until numLabels)
            labels[i] = (bb.get() and 0xFF.toByte()).toInt() // To unsigned

        return labels
    }

    fun getImages(infile: String): ArrayList<Array<IntArray?>> {
        val bb = loadFileToByteBuffer(infile)

        assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.int)

        val numImages = bb.int
        val numRows = bb.int
        val numColumns = bb.int
        val images = ArrayList<Array<IntArray?>>()

        for (i in 0 until numImages)
            images.add(readImage(numRows, numColumns, bb))

        return images
    }

    private fun readImage(numRows: Int, numCols: Int, bb: ByteBuffer): Array<IntArray?> {
        val image = arrayOfNulls<IntArray>(numRows)
        for (row in 0 until numRows)
            image[row] = readRow(numCols, bb)
        return image
    }

    private fun readRow(numCols: Int, bb: ByteBuffer): IntArray {
        val row = IntArray(numCols)
        for (col in 0 until numCols)
            row[col] = (bb.get() and 0xFF.toByte()).toInt() // To unsigned
        return row
    }

    fun assertMagicNumber(expectedMagicNumber: Int, magicNumber: Int) {
        if (expectedMagicNumber != magicNumber) {
            when (expectedMagicNumber) {
                LABEL_FILE_MAGIC_NUMBER -> throw RuntimeException("This is not a label file.")
                IMAGE_FILE_MAGIC_NUMBER -> throw RuntimeException("This is not an image file.")
                else -> throw RuntimeException(
                        format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber))
            }
        }
    }

    /*******
     * Just very ugly utilities below here. Best not to subject yourself to
     * them. ;-)
     */

    fun loadFileToByteBuffer(infile: String): ByteBuffer {
        return ByteBuffer.wrap(loadFile(infile))
    }

    fun loadFile(infile: String): ByteArray {
        try {
            val f = RandomAccessFile(infile, "r")
            val chan = f.channel
            val fileSize = chan.size()
            val bb = ByteBuffer.allocate(fileSize.toInt())
            chan.read(bb)
            bb.flip()
            val baos = ByteArrayOutputStream()
            for (i in 0 until fileSize)
                baos.write(bb.get().toInt())
            chan.close()
            f.close()
            return baos.toByteArray()
        } catch (e: Exception) {
            throw RuntimeException(e)
        }

    }

    fun renderImage(image: Array<IntArray>): String {
        val sb = StringBuffer()

        for (row in image.indices) {
            sb.append("|")
            for (col in 0 until image[row].size) {
                val pixelVal = image[row][col]
                if (pixelVal == 0)
                    sb.append(" ")
                else if (pixelVal < 256 / 3)
                    sb.append(".")
                else if (pixelVal < 2 * (256 / 3))
                    sb.append("x")
                else
                    sb.append("X")
            }
            sb.append("|\n")
        }

        return sb.toString()
    }

    fun repeat(s: String, n: Int): String {
        val sb = StringBuilder()
        for (i in 0 until n)
            sb.append(s)
        return sb.toString()
    }
}