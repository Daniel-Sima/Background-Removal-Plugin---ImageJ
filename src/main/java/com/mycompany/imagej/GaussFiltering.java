/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.mycompany.imagej;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.process.ByteProcessor;
import ij.process.ShortProcessor;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.ejml.simple.SimpleMatrix;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

/**
 * This example illustrates how to create an ImageJ {@link Command} plugin.
 * <p>
 * The code here is a simple Gaussian blur using ImageJ Ops.
 * </p>
 * <p>
 * You should replace the parameter fields with your own inputs and outputs,
 * and replace the {@link run} method implementation with your own logic.
 * </p>
 */
@Plugin(type = Command.class, menuPath = "Plugins>Gauss Filtering")
public class GaussFiltering<T extends RealType<T>> implements Command {
    //
    // Feel free to add more parameters here...
    //
    @Parameter
    private Dataset currentData;
    @Parameter
    private UIService uiService;
    @Parameter
    private OpService opService;
    @Override
    public void run() {
        final Img<T> image = (Img<T>) currentData.getImgPlus();

        //
        // Enter image processing code here ...
        // The following is just a Gauss filtering example
        //
        final double[] sigmas = {1.0, 3.0, 5.0};

        List<RandomAccessibleInterval<T>> results = new ArrayList<>();

        for (double sigma : sigmas) {
            results.add(opService.filter().gauss(image, sigma));
        }

        // display result
        for (RandomAccessibleInterval<T> elem : results) {
            uiService.show(elem);
        }
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------------------------------*/
    /**
     * This main function serves for development purposes.
     * It allows you to run the plugin immediately out of
     * your integrated development environment (IDE).
     *
     * @param args whatever, it's ignored
     * @throws Exception
     */
    private static double tau = 7;
    private static int k = 2;
    private static int rank = 3;
    private static double tol = 0.001;
    private static int power = 5;
    private static MODE mode = MODE.SOFT;
    private static int height, width, nbSlices, type, dynamicRange = 8;
    public static void main(final String... args) throws Exception {

        /*
        load file with project properties
         */
//        Properties properties = new Properties();
//        try (InputStream inputStream = GaussFiltering.class.getClassLoader().getResourceAsStream("application.properties")) {
//            properties.load(inputStream);
//        } catch (IOException e) {
//            // handle exception
//        }

        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();

        // affiche l'interface utilisateur pour acceder aux functionnalites
        ij.ui().showUI();

        // for the future
        /*  ask the user for a file to open */
        final File file = ij.ui().chooseFile(null, "open");


        /*
        use constantValue from src/main/resources/application.properties for DEV MODE
         */
//        String constantValue = properties.getProperty("constant.variable.filename");
//        final File file = new File(constantValue);


        if (file != null) {

            ImagePlus imp = importImage(file, 0);

            generateInterface();

            long startTime = System.nanoTime();

            /** originalImg
             * It is matrix[rows*cols] where:
             *   rows = stackSize
             *   cols = width*height
             * This means that each line represents one of the layers of the tif image
             *
             */
            DenseMatrix64F originalImg = constructMatrix2(imp.getStack(), type);

            //transposing
            DenseMatrix64F origImgTransposed = new DenseMatrix64F(originalImg.numCols, originalImg.numRows);
            CommonOps.transpose(originalImg, origImgTransposed);

            //taking the transposed image matrix for future operations
            originalImg = origImgTransposed;

            /*
             * SVD decomposition
             * svdResult = ArrayList<X, Y, s>
             * X = Unitary matrix having left singular vectors as columns.
             * Y = Unitary matrix having right singular vectors as rows.
             * s = Diagonal matrix with singular values.
             */
            ArrayList<DenseMatrix64F> svdResult = svdDecomposition(originalImg);

            DenseMatrix64F X = svdResult.get(0);
            DenseMatrix64F Y = svdResult.get(1);
            DenseMatrix64F s = svdResult.get(2);

            // X = X * s
            DenseMatrix64F sX = new DenseMatrix64F(X.numRows, s.numCols);
            CommonOps.mult(X, s, sX);
            X = sX;

            //L = X * Y
            DenseMatrix64F L = new DenseMatrix64F(X.numRows, Y.numCols);
            CommonOps.mult(X, Y, L);

            //S = originalImg - L
            DenseMatrix64F S = new DenseMatrix64F(originalImg.numRows, originalImg.numCols);
            CommonOps.sub(originalImg, L, S);

            //thresholding
            DenseMatrix64F thresholdS = threshold(S, tau, String.valueOf(mode));

            //error calculation
            int rankk = (int) Math.round((double) rank / k);
            DenseMatrix64F error = new DenseMatrix64F(rank * power, 1);

            //T = S - thresholdS
            DenseMatrix64F T = new DenseMatrix64F(S.numRows, S.numCols);
            CommonOps.sub(S, thresholdS, T);

            double normD = NormOps.normF(originalImg);
            double normT = NormOps.normF(T);

            error.set(0, normT / normD);

            int iii = 1;
            boolean stop = false;
            double alf;

            for (int i = 1; i < rankk + 1; i++) {
                i = i - 1;
                int rrank = rank;
                alf = 0;
                double increment = 1;

                if (iii == power * (i - 2) + 1) {
                    iii = iii + power;
                }

                for (int j = 1; j < power + 1; j++) {

                    /*
                     *  X update
                     */

                    //Y transpose
                    DenseMatrix64F transposedY = new DenseMatrix64F(Y.numCols, Y.numRows);
                    CommonOps.transpose(Y, transposedY);

                    //X = abs(L * transposedY)
                    DenseMatrix64F LY = new DenseMatrix64F(L.numRows, transposedY.numCols);
                    CommonOps.mult(L, transposedY, LY);
                    X = LY;
                    for (int k = 0; k < X.numCols; k++) {
                        for (int l = 0; l < X.numRows; l++) {
                            X.set(l, k, Math.abs(X.get(l, k)));
                        }
                    }

                    /* Do a QR decomposition */
                    /* Possible qu'ici la decomposition se fasse mal */

                    //X = qr.Q
                    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X.numRows, X.numCols);
                    qr.decompose(X);
                    qr.getQ(X, true);
//                    X = QRFactorisation_Q(X, j);

                    /*
                     *   Y update
                     */
                    //Y = transposedX * L
                    DenseMatrix64F transposedX = new DenseMatrix64F(X.numCols, X.numRows);
                    CommonOps.transpose(X, transposedX);
                    CommonOps.mult(transposedX, L, Y);

                    //L = X * Y
                    CommonOps.mult(X, Y, L);

                    /*
                     *  S update
                     */
                    //T = originalImg - L
                    CommonOps.sub(originalImg, L, T);
                    //thresholding
                    S = threshold(T, tau, String.valueOf(mode));

                    // Error, stopping criteria
                    //T = T - S
                    CommonOps.sub(T, S, T);

                    int ii = iii + j - 1;

                    normT = NormOps.normF(T);
                    error.set(ii, normT / normD);

                    if (error.get(ii) < tol) {
                        stop = true;
                        System.out.println("Stop");
                        break;
                    }

                    if (rrank != rank) {
                        System.out.println("rrank != rank");
                        rank = rrank;
                    }

                    // Adjust alf
                    double ratio = error.get(ii) / error.get(ii - 1);

                    if (ratio >= 1.1) {
                        increment = Math.max(0.1 * alf, 0.1 * increment);
                        error.set(ii, error.get(ii - 1));
                        alf = 0;
                    } else if (ratio > 0.7) {
                        increment = Math.max(increment, 0.25 * alf);
                        alf = alf + increment;
                    }

                    /*
                     *   Update of L
                     */
                    //T = (1 + alf) * T
                    CommonOps.scale(1 + alf, T, T);
                    //L = L * T
                    CommonOps.add(L, T, L);

                    // Add corest AR
                    if (j > 8) {
                        double mean = mean(error, ii - 7, ii);
                        if (mean > 0.92) {
                            iii = ii;
                            int YCol = Y.numCols;
                            int XRow = X.numRows;
                            if ((YCol - XRow) >= k) {
                                CommonOps.extract(Y, 0, XRow - 1, 0, Y.numCols, Y, 0, 0);
                            }
                            break;
                        }
                    }
                }

                if (stop) {
                    break;
                }

//            	AR
                if (i + 1 < rankk) {
                    Random r = new Random();
                    DenseMatrix64F RR = new DenseMatrix64F(k, originalImg.numRows);
                    for (int x = 0; x < k; x++) {
                        for (int z = 0; z < originalImg.numRows; z++) {
                            RR.set(x, z, r.nextGaussian());
//            	    		RR.set(x, z, 1.0);
                        }
                    }

                    //v = RR * L
                    DenseMatrix64F v = new DenseMatrix64F(RR.numRows, L.numCols);
                    CommonOps.mult(RR, L, v);

                    //Y = combine(Y, v)
                    DenseMatrix64F newY = new DenseMatrix64F(Y.numRows * 2, Y.numCols);
                    for (int p = 0; p < Y.numRows; p++) {
                        for (int q = 0; q < Y.numCols; q++) {
                            newY.set(p, q, Y.get(p, q));
                        }
                    }
                    for (int p = 0; p < v.numRows; p++) {
                        for (int q = 0; q < v.numCols; q++) {
                            newY.set(p + Y.numRows, q, v.get(p, q));
                        }
                    }
                    Y = newY;
                }
                i++;
            }

            //L = X * Y
            CommonOps.mult(X, Y, L);

            if (originalImg.numRows > originalImg.numCols) {
                DenseMatrix64F transposedL = new DenseMatrix64F(L.numCols, L.numRows);
                CommonOps.transpose(L, transposedL);
                L = transposedL;

                DenseMatrix64F transposedS = new DenseMatrix64F(S.numCols, S.numRows);
                CommonOps.transpose(S, transposedS);
                S = transposedS;

                DenseMatrix64F transposedOriginalImg = new DenseMatrix64F(originalImg.numCols, originalImg.numRows);
                CommonOps.transpose(originalImg, transposedOriginalImg);
                originalImg = transposedOriginalImg;
            }

            /* Noise: G = originalImg - L - S */
            DenseMatrix64F G = new DenseMatrix64F(originalImg.numRows, originalImg.numCols);
            CommonOps.sub(originalImg, L, G);
            CommonOps.sub(G, S, G);

            double[][] A2 = matrix2Array2(originalImg);
            double[][] L2 = matrix2Array2(L);
            double[][] S2 = matrix2Array2(S);
            double[][] G2 = matrix2Array2(G);

            ImagePlus original = new ImagePlus("Original Image", constructImageStack2(
                    A2, type));
            ImagePlus im = new ImagePlus("Background Image", constructImageStack2(
                    L2, type));
            ImagePlus im2 = new ImagePlus("Sparse Image", constructImageStack2(
                    S2, type));
            /* Construction du stack d'images pour le bruit (noise) */
            ImagePlus noise = new ImagePlus("Noise Image", constructImageStack2(G2, type));

            im.show();
            im2.show();
            original.show();
            noise.show();		// Affichage du bruit (noise)

            long endTime = System.nanoTime();
            long duration = endTime - startTime;
            long durationInMilliseconds = duration / 1000000;
            double durationInSeconds = duration / 1000000000.0;
            System.out.println("Execution time in nanoseconds: " + duration);
            System.out.println("Execution time in milliseconds: " + durationInMilliseconds);
            System.out.println("Execution time in seconds: " + durationInSeconds);

            // show the image
            //ij.ui().show(dataset);

            // invoke the plugin
            //ij.command().run(GaussFiltering.class, true);
        }
    }

    /*-----------------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static ImagePlus importImage(File file, int maxFrames) {

        checkFileExtension(file);

        //load image in object
        ImagePlus imp = IJ.openImage(file.getPath());  // image

        int stackSize = imp.getStackSize();
        int ImgHeight = imp.getHeight();
        int ImgWidth = imp.getWidth();

        // FIXME probleme s'il y a qu'une seule frame ?
        if (imp.getStackSize() < 2) {
            IJ.error("Stack required");
            throw new RuntimeException("Stack required");
        } else { // si plusieurs frames
            nbSlices = maxFrames == 0 ? stackSize : maxFrames;
            if (imp.getType() == ImagePlus.GRAY8) type = 8;
            else if (imp.getType() == ImagePlus.GRAY16) type = 16;
            else {
                IJ.error("Image type not supported ( only GRAY8 and GRAY16 )");
                throw new RuntimeException("Image type not supported ( only GRAY8 and GRAY16 )");
            }
        }

        height = ImgHeight;
        width = ImgWidth;

        return imp;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void checkFileExtension(File file) {
        String fileExtension = getFileExtension(file);

        if (fileExtension.equals("tif") || fileExtension.equals("tiff")) {
            System.out.println("TIF stack loading OK");
        } else {
            throw new RuntimeException("The file extension should be .tif, .tiff");
        }
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void generateInterface() {
        String[] choice = {"soft", "hard"};
        GenericDialog d = new GenericDialog("Threshold");
        d.addNumericField("tau:", tau, 2);
        d.addChoice("soft or hard thresholding", choice, choice[0]);
        d.showDialog();
        if (d.wasCanceled()) return;

        // recuperation de la valeur saisie par l'user
        tau = d.getNextNumber();
        int c = d.getNextChoiceIndex();
        if (c == 0) mode = MODE.SOFT;
        else mode = MODE.HARD;
        if (d.invalidNumber()) {
            IJ.error("Invalid parameters");
            return;
        }
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static String getFileExtension(File file) {
        String fileName = file.getName();
        if (fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0)
            return fileName.substring(fileName.lastIndexOf(".") + 1);
        else return "";
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double[][][] constructMatrix(ImageStack stack, int bit) {
        System.out.println(nbSlices);
        System.out.println(height);
        System.out.println(width);
        double[][][] matrix = new double[nbSlices][width][height];
        if (bit == 8) {
            for (int z = 0; z < nbSlices; z++) {
                ByteProcessor bp = (ByteProcessor) stack.getProcessor(z + 1);
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        matrix[z][i][j] = bp.getPixelValue(i, j);
            }
        }
        if (bit == 16) {
            for (int z = 0; z < nbSlices; z++) {
                ShortProcessor bp = (ShortProcessor) stack.getProcessor(z + 1);
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        matrix[z][i][j] = bp.getPixelValue(i, j);

            }
        }
        return matrix;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static DenseMatrix64F constructMatrix2(ImageStack stack, int bit) {
        double[][] matrix = new double[nbSlices][width * height];

        if (bit == 8) {
            for (int z = 0; z < nbSlices; z++) {
                ByteProcessor bp = (ByteProcessor) stack.getProcessor(z + 1);
                int index = 0;
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        matrix[z][index++] = bp.getPixelValue(j, i);
                    }
                }
            }
        }
        return new DenseMatrix64F(matrix);
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static ArrayList<DenseMatrix64F> svdDecomposition(DenseMatrix64F originalImg) {
        ArrayList<DenseMatrix64F> result = new ArrayList<>(3);

        org.ejml.factory.SingularValueDecomposition<DenseMatrix64F> svd =
                DecompositionFactory.svd(originalImg.numRows, k, true, true, true);

        svd.decompose(originalImg);
        DenseMatrix64F U = svd.getU(null, false);
        DenseMatrix64F W = svd.getW(null);
        DenseMatrix64F V = svd.getV(null, false);
        double[] allSingVals = svd.getSingularValues();
        double[] kSingVals = new double[k];
        for (int i = 0; i < k; i++) {
            kSingVals[i] = allSingVals[k - 1 - i];
        }

        //taking k first columns as having biggest singular values
        DenseMatrix64F X = new DenseMatrix64F(U.numRows, k);
        CommonOps.extract(U, 0, U.numRows, 0, k, X, 0, 0);

        //putting X to ascending order
        DenseMatrix64F invertedX = new DenseMatrix64F(X.numRows, X.numCols);
        for (int e = X.numCols - 1, i = 0; i < X.numCols; i++, e--) {
            for (int j = 0; j < X.numRows; j++) {
                invertedX.set(j, e, X.get(j, i));
            }
        }
        X = invertedX;

        // mult par -1 pour changer de signe pour avoir les mêmes val qu'en python psq jsp pq y avait un -
        SimpleMatrix negatif = SimpleMatrix.identity(k);
        negatif.set(k-1, k-1, -1);
        DenseMatrix64F neg = negatif.getMatrix();

        DenseMatrix64F negX = new DenseMatrix64F(X.numRows, neg.numCols);
        CommonOps.mult(X, neg, negX);
        X = negX;

        result.add(X);

        //taking k first columns as having biggest singular values
        DenseMatrix64F Y = new DenseMatrix64F(V.numRows, k);
        CommonOps.extract(V, 0, V.numRows, 0, k, Y, 0, 0);

        //putting Y to ascending order
        DenseMatrix64F invertedY = new DenseMatrix64F(Y.numRows, Y.numCols);
        for (int e = Y.numCols - 1, i = 0; i < Y.numCols; i++, e--) {
            for (int j = 0; j < Y.numRows; j++) {
                invertedY.set(j, e, Y.get(j, i));
            }
        }

        // mult par -1 pour changer de signe pour avoir les mêmes val qu'en python psq jsp pq y avait un -
        DenseMatrix64F negY = new DenseMatrix64F(invertedY.numRows, neg.numCols);
        CommonOps.mult(invertedY, neg, negY);
        Y = negY;

        DenseMatrix64F YTrans = new DenseMatrix64F(Y.numCols, Y.numRows);
        CommonOps.transpose(Y, YTrans);

        Y = YTrans;
        result.add(Y);

        //getting submatrix of SingularValues in inverted order
        SimpleMatrix s = SimpleMatrix.diag(kSingVals);
        result.add(s.getMatrix());

        return result;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static ImageStack constructImageStack(double[][][] matrix, int bit) {
        ImageStack newStack = new ImageStack(width, height);
        for (int z = 0; z < nbSlices; z++) {
            ByteProcessor bp = new ByteProcessor(width, height);

            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    bp.putPixel(i, j, (int) matrix[z][i][j]);
            newStack.addSlice(bp);
        }

        return newStack;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static ImageStack constructImageStack2(double[][] matrix, int bit) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        // trouver la valeur minimale et la valeur maximale
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                min = Math.min(min, matrix[i][j]);
                max = Math.max(max, matrix[i][j]);
            }
        }

        // normaliser les données
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = ((matrix[i][j] - min) / (max - min)) * (Math.pow(2, dynamicRange) - 1);
            }
        }

        ImageStack newStack = new ImageStack(width, height);
        for (int z = 0; z < nbSlices; z++) {
            byte[] pixels = new byte[width * height];
            if (dynamicRange == 8) {
                for (int i = 0; i < pixels.length; i++) {
                    pixels[i] = (byte) Math.round(matrix[z][i]);
                }
                ByteProcessor bp = new ByteProcessor(width, height, pixels);

                newStack.addSlice(bp);
            }
//          } else if (dynamicRange == 16) {
//          short[][] shortData = new short[S2.length][S2[0].length];
//          for (int i = 0; i < S2.length; i++) {
//              for (int j = 0; j < S2[i].length; j++) {
//                  shortData[i][j] = (short) Math.round(S2[i][j]);
//              }
//          }
//          S2 = shortData;
            else {
                System.out.println("The dynamic range should be equal to 8 or 16 (bits)");
            }
        }

        return newStack;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double[][] matrix2Array2(DenseMatrix64F matrix) {
        double[][] array = new double[matrix.numRows][matrix.numCols];
        for (int r = 0; r < matrix.numRows; r++) {
            for (int c = 0; c < matrix.numCols; c++) {
                array[r][c] = matrix.get(r, c);
            }
        }
        return array;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static SimpleMatrix thresholding(SimpleMatrix S) {
        for (int i = 0; i < S.numRows(); i++) {
            for (int j = 0; j < S.numCols(); j++) {
                double val = S.get(i, j);
                if (val > tau) {
                    S.set(i, j, val - tau);
                } else if (val < -tau) {
                    S.set(i, j, val + tau);
                } else {
                    S.set(i, j, 0);
                }
            }
        }
        return S;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static DenseMatrix64F threshold(DenseMatrix64F data, double tau, String mode) {
        int rows = data.numRows;
        int cols = data.numCols;
        DenseMatrix64F result = new DenseMatrix64F(rows, cols);
        if (mode.equals("SOFT")) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data.get(i, j);
                    if (Math.abs(val) < tau) {
                        result.set(i, j, 0);
                    } else {
                        result.set(i, j, (val / Math.abs(val) * Math.max(Math.abs(val) - tau, 0)));
                    }
                }
            }
        } else if (mode.equals("HARD")) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data.get(i, j);
                    if (Math.abs(val) < tau) {
                        result.set(i, j, 0);
                    } else {
                        result.set(i, j, val);
                    }
                }
            }
        } else {
            System.out.println("mode not supported");
            throw new RuntimeException("mode not supported");
        }
        return result;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double mean(DenseMatrix64F matrix, int start, int end) {
        double sum = 0;
        int count = 0;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < matrix.numCols; j++) {
                sum += matrix.get(i, j);
                count++;
            }
        }
        return sum / count;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double[][][] matrix2Array(SimpleMatrix matrix) {
        double[][][] array = new double[nbSlices][width][height];
        for (int r = 0; r < matrix.numRows(); r++) {
            for (int c = 0; c < matrix.numCols(); c++) {
                int row = (c % (width * height)) / height;
                int col = (c % (width * height)) % height;
                array[r][row][col] = matrix.get(r, c);
            }
        }
        return array;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double[][][] matrix2Array3(SimpleMatrix matrix) {
        double[][][] array = new double[nbSlices][width][height];
        for (int r = 0; r < matrix.numRows(); r++) {
            for (int c = 0; c < matrix.numCols(); c++) {
                int row = (c % (width * height)) / height;
                int col = (c % (width * height)) % height;
                array[r][row][col] = matrix.get(r, c);
            }
        }
        return array;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static DenseMatrix64F QRFactorisation_Q(DenseMatrix64F matrix, int negatif) {
        if (matrix.numRows < matrix.numCols) {
            System.out.println("Il doit y avoir plus de lignes que de colonnes");
            throw new RuntimeException("Il doit y avoir plus de lignes que de colonnes");
        }

        boolean multiplier = false;
        if (negatif <= 2) {
            multiplier = true;
        }

        DenseMatrix64F Q = new DenseMatrix64F(matrix.numRows, matrix.numCols);
        Q.zero();

        DenseMatrix64F W, Wbis, Qbis, aux;
        for (int i = 0; i < Q.numCols; i++) {
            W = new DenseMatrix64F(matrix.numRows, 1);
            CommonOps.extract(matrix, 0, matrix.numRows, i, i + 1, W, 0, 0);
            Wbis = W;
            for (int j = 0; j < i; j++) {
                Qbis = new DenseMatrix64F(Q.numRows, 1);
                CommonOps.extract(Q, 0, Q.numRows, j, j + 1, Qbis, 0, 0);
                double dotProduct = 0;
                for (int p = 0; p < Wbis.numRows; p++) {
                    dotProduct += Wbis.get(p, 0) * Qbis.get(p, 0);
                }
                CommonOps.scale(dotProduct, Qbis, Qbis);
                CommonOps.sub(W, Qbis, W);
            }
            aux = new DenseMatrix64F(W.numRows, W.numCols);
            CommonOps.divide(NormOps.normF(W), W, aux);
            double[] res = new double[aux.numRows];
            for (int k = 0; k < aux.numRows; k++) {
                if ((multiplier) && (i == Q.numCols - 1)) {
                    res[k] = aux.get(k, 0) * -1;
                    continue;
                }
                res[k] = aux.get(k, 0);
            }

            for (int row = 0; row < Q.numRows; row++) {
                Q.set(row, i, res[row]);
            }
        }

        return Q;

    }
    private enum MODE {SOFT, HARD;}
}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
