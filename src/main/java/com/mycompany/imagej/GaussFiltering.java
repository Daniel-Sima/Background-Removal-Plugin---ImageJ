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
import net.imglib2.type.numeric.RealType;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.util.ArrayList;
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
        execute();
    }

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
    private enum MODE {SOFT, HARD;}
    private static int height, width, nbSlices, nbSlicesPreview, type, dynamicRange = 8;
    public static void main(final String... args) throws Exception {

        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();

        // affiche l'interface utilisateur pour acceder aux functionnalites
        ij.ui().showUI();

        // for the future
        /*  ask the user for a file to open */
        final File file = ij.ui().chooseFile(null, "open");

        if (file != null) {
            checkFileExtension(file);

            //load image in object
            ImagePlus imp = IJ.openImage(file.getPath());  // image
            importImage(imp, 0);

            process(imp);
        }
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void execute() {
        ImagePlus imp = IJ.getImage();
        importImage(imp, 0);
        process(imp);
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void process(ImagePlus imp) {
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
//                    DenseMatrix64F X2 = X.getMatrix();
//                    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X2.numRows, X2.numCols);
//                    qr.decompose(X2);
//                    qr.getQ(X2, true);
//                    X = SimpleMatrix.wrap(X2);

                X = QRFactorisation_Q(X, j);
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
                        int YCol = Y.numCols();
                        int XRow = X.numRows();
                        if ((YCol - XRow) >= k) {
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

        /* Noise: G = originalImg - L - S */
        SimpleMatrix G = (originalImg.minus(L)).minus(S);

        double[][] A2 = matrix2Array(originalImg);
        double[][] L2 = matrix2Array(L);
        double[][] S2 = matrix2Array(S);
        double[][] G2 = matrix2Array(G);

        ImagePlus original = new ImagePlus(" Original Image ", constructImageStack(
                A2, type));
        ImagePlus im = new ImagePlus(" Background Image ", constructImageStack(
                L2, type));
        ImagePlus im2 = new ImagePlus(" Sparse Image ", constructImageStack(
                S2, type));
        /* Construction du stack d'images pour le bruit (noise) */
        ImagePlus noise = new ImagePlus("Noise Image", constructImageStack(G2, type));


        im.show();
        im2.show();
        original.show();
        noise.show();        // Affichage du bruit (noise)

        long endTime = System.nanoTime();
        long duration = endTime - startTime;
        long durationInMilliseconds = duration / 1000000;
        double durationInSeconds = duration / 1000000000.0;
        System.out.println("Execution time in nanoseconds: " + duration);
        System.out.println("Execution time in milliseconds: " + durationInMilliseconds);
        System.out.println("Execution time in seconds: " + durationInSeconds);
    }

    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void importImage(ImagePlus imp, int maxFrames) {

        int stackSize = imp.getStackSize();
        int ImgHeight = imp.getHeight();
        int ImgWidth = imp.getWidth();

        // FIXME probleme s'il y a qu'une seule frame ?
        if (imp.getStackSize() < 2) {
            IJ.error("Stack required");
            throw new RuntimeException("Stack required");
        } else {
            nbSlicesPreview = Math.min(stackSize, 100);
            nbSlices = stackSize;
            if (imp.getType() == ImagePlus.GRAY8) type = 8;
            else if (imp.getType() == ImagePlus.GRAY16) type = 16;
            else {
                IJ.error("Image type not supported ( only GRAY8 and GRAY16 )");
            }
        }

        height = ImgHeight;
        width = ImgWidth;
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
    public static String getFileExtension(File file) {
        String fileName = file.getName();
        if (fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0)
            return fileName.substring(fileName.lastIndexOf(".") + 1);
        else return "";
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void generateInterface() {
        String[] choice = {"soft", "hard"};
        GenericDialog d = new GenericDialog("Low Rank and Sparse tool");
        d.addNumericField("tau:", tau, 2);
        d.addChoice("soft or hard thresholding", choice, choice[0]);
        d.showDialog();
        if (d.wasCanceled()) return;
        // recuperation de la valeur saisie par l'user
        tau = d.getNextNumber();
        String filePath = d.getNextString();
        int c = d.getNextChoiceIndex();
        if (c == 0) mode = MODE.SOFT;
        else mode = MODE.HARD;
        if (d.invalidNumber()) {
            IJ.error("Invalid parameters");
            return;
        }
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
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
        // mult par -1 pour changer de signe pour avoir les mêmes val qu'en python psq jsp pq y avait un -
        SimpleMatrix negatif = SimpleMatrix.identity(k);
        negatif.set(k-1, k-1, -1);
        invertedX = invertedX.mult(negatif);

        X = invertedX;

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
        invertedY = invertedY.mult(negatif);
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
    /*-----------------------------------------------------------------------------------------------------------------------*/

    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static ImageStack constructImageStack(double[][] matrix, int bit) {
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
    private static double[][] matrix2Array(SimpleMatrix matrix) {
        double[][] array = new double[matrix.numRows()][matrix.numCols()];
        for (int r = 0; r < matrix.numRows(); r++) {
            for (int c = 0; c < matrix.numCols(); c++) {
                array[r][c] = matrix.get(r, c);
            }
        }
        return array;
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
            System.out.println("mode not supported");
            throw new RuntimeException("mode not supported");
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

    private static SimpleMatrix QRFactorisation_Q(SimpleMatrix matrix, int negatif) {
        if (matrix.numRows() < matrix.numCols()) {
            System.out.println("Il doit y avoir plus de lignes que de colonnes");
            throw new RuntimeException("Il doit y avoir plus de lignes que de colonnes");
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

}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
