/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.mycompany.imagej;
//
//import java.util.Locale;
//
//import org.apache.commons.math3.linear.Array2DRowRealMatrix;
//import org.apache.commons.math3.linear.QRDecomposition;
//import org.apache.commons.math3.linear.RealMatrix;

import org.la4j.Matrix;
import org.la4j.decomposition.QRDecompositor;


import net.imagej.Dataset;
//import cern.colt.matrix.impl.DenseDoubleMatrix2D;
//import cern.colt.matrix.linalg.QRDecomposition;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.RealType;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

//import Jama.Matrix;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.gui.StackWindow;
import ij.io.Opener;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.ejml.alg.dense.decomposition.qr.QRDecompositionHouseholderColumn;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DenseMatrix64F;
import org.ejml.dense.row.SingularOps_DDRM;
import org.ejml.dense.row.decomposition.qr.QRDecompositionHouseholderColumn_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.DecompositionInterface;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.factory.QRDecomposition;
//import org.ejml.factory.QRDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SingularOps;
import org.ejml.simple.SimpleBase;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

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
    @Parameter
    private Dataset currentData;
    @Parameter
    private UIService uiService;
    @Parameter
    private OpService opService;

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
            SimpleMatrix originalImg = constructMatrix(imp.getStack(), type);

            //transposing
            originalImg = originalImg.transpose();

            /*
             * SVD decomposition
             * svdResult = ArrayList<X, Y, s>
             * X = Unitary matrix having left singular vectors as columns.
             * Y = Unitary matrix having right singular vectors as rows.
             * s = Diagonal matrix with singular values.
             */
            ArrayList<SimpleMatrix> svdResult = svdDecomposition(originalImg);

            SimpleMatrix X = svdResult.get(0);
            SimpleMatrix Y = svdResult.get(1);
            SimpleMatrix s = svdResult.get(2);

            // X = X * s
            X = X.mult(s);

            //L = X * Y
            SimpleMatrix L = X.mult(Y);

            //S = originalImg - L
            SimpleMatrix S = originalImg.minus(L);

            //thresholding
            S = threshold(S, tau, String.valueOf(mode));

            //error calculation
            int rankk = (int) Math.round((double) rank / k);
            SimpleMatrix error = new SimpleMatrix(rank * power, 1);

            //T = S - thresholdS
            SimpleMatrix T = (originalImg.minus(L)).minus(S);

            double normD = originalImg.normF();
            double normT = T.normF();

            error.set(0, normT / normD);

            int iii = 1;
            boolean stop = false;
            double alf = 0;

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

                    //X = abs(L * transposedY)
                    X = (L.mult(Y.transpose()));
                    for (int k = 0; k < X.numCols(); k++) {
                        for (int l = 0; l < X.numRows(); l++) {
                            X.set(l, k, Math.abs(X.get(l, k)));
                        }
                    }

                    /* Do a QR decomposition */
                    DenseMatrix64F X2 = X.getMatrix();
                    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X2.numRows, X2.numCols);
                    qr.decompose(X2);
                    qr.getQ(X2, true);
                    X = SimpleMatrix.wrap(X2);

                    //X = QRFactorisation_Q(X, j);
//
                    /*
                     *   Y update
                     */
                    //Y = transposedX * L
                    Y = (X.transpose()).mult(L);

                    //L = X * Y
                    L = X.mult(Y);

                    /*
                     *  S update
                     */
                    //T = originalImg - L
                    T = originalImg.minus(L);
                    //thresholding
//                    S = thresholding(T);
                    S = threshold(T, tau, String.valueOf(mode));

                    // Error, stopping criteria
                    //T = T - S
                    T = T.minus(S);

                    int ii = iii + j - 1;

                    normT = T.normF();

                    error.set(ii, normT / normD);

                    if (error.get(ii) < tol) {
                        stop = true;
                        break;
                    }

                    if (rrank != rank) {
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
                    //L = L + T
                    L = L.plus(T.scale(1 + alf));

                    // Add corest AR
                    if (j > 8) {
                        double mean = mean(error, ii - 7, ii);
                        if (mean > 0.92) {
                            iii = ii;
                            int YRow = Y.numCols();
                            int XRow = X.numRows();
                            if ((YRow - XRow) >= k) {
                                Y = Y.extractMatrix(0, XRow - 1, 0, Y.numCols());
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
                    SimpleMatrix RR = new SimpleMatrix(k, originalImg.numRows());
                    for (int x = 0; x < k; x++) {
                        for (int z = 0; z < originalImg.numRows(); z++) {
                            RR.set(x, z, r.nextGaussian());
                            //RR.set(x, z, 1.0);
                        }
                    }
                    //v = RR * L
                    SimpleMatrix v = RR.mult(L);
                    //Y = combine(Y, v)
                    Y = Y.combine(2, 0, v);
                }
                i++;

            }

            //L = X * Y
            L = X.mult(Y);

            if (originalImg.numRows() > originalImg.numCols()) {

                L = L.transpose();
                S = S.transpose();
                originalImg = originalImg.transpose();
            }


            double[][] A2 = matrix2Array2(originalImg);
            double[][] L2 = matrix2Array2(L);
            double[][] S2 = matrix2Array2(S);

            ImagePlus original = new ImagePlus(" Original Image ", constructImageStack2(
                    A2, type));
            ImagePlus im = new ImagePlus(" Background Image ", constructImageStack2(
                    L2, type));
            ImagePlus im2 = new ImagePlus(" Sparse Image ", constructImageStack2(
                    S2, type));


            im.show();
            im2.show();
            original.show();

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
            nbSlices = stackSize;
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
    /*
     * Code from last year's project
     */
    private static SimpleMatrix constructMatrix(ImageStack stack, int bit) {
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

        return new SimpleMatrix(matrix);
    }

    private static ArrayList<SimpleMatrix> svdDecomposition(SimpleMatrix originalImg) {
        ArrayList<SimpleMatrix> result = new ArrayList<>(3);

        SimpleMatrix U = null, W = null, V = null, check = null;
        SimpleSVD<SimpleMatrix> svd = originalImg.svd(true); // compacte pour aller plus vite ...

        U = svd.getU();
        W = svd.getW();
        V = svd.getV();

        //taking k first columns as having biggest singular values
        SimpleMatrix X = U.extractMatrix(0, U.numRows(), 0, k); // CMMT

        //putting X to ascending order
        SimpleMatrix invertedX = new SimpleMatrix(X.numRows(), X.numCols());
        for (int e = X.numCols() - 1, i = 0; i < X.numCols(); i++, e--) {
            for (int j = 0; j < X.numRows(); j++) {
                invertedX.set(j, e, X.get(j, i));
            }
        }

        result.add(X);

        //taking k first columns as having biggest singular values
        SimpleMatrix Y = svd.getV().extractMatrix(0, V.numCols(), 0, k); // CMMT

        //putting Y to ascending order
        SimpleMatrix invertedY = new SimpleMatrix(Y.numRows(), Y.numCols());
        for (int e = Y.numCols() - 1, i = 0; i < Y.numCols(); i++, e--) {
            for (int j = 0; j < Y.numRows(); j++) {
                invertedY.set(j, e, Y.get(j, i));
            }
        }

        invertedY = invertedY.transpose();
        Y = invertedY;
        result.add(Y);

        //getting submatrix of SingularValues in inverted order
        SimpleMatrix sVals = W.extractMatrix(0, k, 0, k);
        SimpleMatrix sDiag = sVals.extractDiag();

        double[] valS = new double[sDiag.numRows()];
        for (int j = 0, i = sDiag.numRows() - 1; i >= 0; i--, j++) {
            valS[j] = sDiag.get(i, 0);
        }
        SimpleMatrix s = SimpleMatrix.diag(valS);
        result.add(s);

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
    private static ImageStack constructImageStack3(double[][] matrix, int bit) {
        ImageStack newStack = new ImageStack(width, height);
        for (int z = 0; z < nbSlices; z++) {
            byte[] pixels = new byte[width * height];
            for (int i = 0; i < pixels.length; i++) {
                pixels[i] = (byte) (matrix[z][i] + 0.5);
            }
            ByteProcessor bp = new ByteProcessor(width, height, pixels);

            newStack.addSlice("slice " + (z + 1), bp);
        }

        return newStack;
    }

    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double[][] matrix2Array2(SimpleMatrix matrix) {
        double[][] array = new double[matrix.numRows()][matrix.numCols()];
        for (int r = 0; r < matrix.numRows(); r++) {
            for (int c = 0; c < matrix.numCols(); c++) {
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
    public static SimpleMatrix threshold(SimpleMatrix data, double tau, String mode) {
        int rows = data.numRows();
        int cols = data.numCols();
        SimpleMatrix result = new SimpleMatrix(rows, cols);
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
            IJ.error("mode not supported");
            return null;
        }
        return result;
    }

    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static double mean(SimpleMatrix matrix, int start, int end) {
        double sum = 0;
        int count = 0;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                sum += matrix.get(i, j);
                count++;
            }
        }
        return sum / count;
    }

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

    private static SimpleMatrix QRFactorisation_Q(SimpleMatrix matrix, int negatif) {
        if (matrix.numRows() < matrix.numCols()) {
            System.out.println("Il doit y avoir plus de lignes que de colonnes");
            return null;
        }

        boolean multiplier = false;
        if (negatif <= 2) {
            multiplier = true;
        }

        SimpleMatrix Q = new SimpleMatrix(matrix.numRows(), matrix.numCols());
        Q.zero();

        SimpleMatrix W, Wbis, Qbis, aux;
        for (int i = 0; i < Q.numCols(); i++) {
            W = matrix.extractVector(false, i);
            Wbis = W;
            for (int j = 0; j < i; j++) {
                Qbis = Q.extractVector(false, j);
                W = W.minus(Qbis.scale(Wbis.dot(Qbis)));
            }
            aux = W.divide(W.normF());
            double[] res = new double[aux.numRows()];
            for (int k = 0; k < aux.numRows(); k++) {
                if ((multiplier) && (i == Q.numCols() - 1)) {
                    res[k] = aux.get(k, 0) * -1;
                    continue;
                }
                res[k] = aux.get(k, 0);
            }

            Q.setColumn(i, 0, res);
        }

        return Q;

    }

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


    private enum MODE {SOFT, HARD;}

}
