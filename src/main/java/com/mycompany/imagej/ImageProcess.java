package com.mycompany.imagej;


import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Random;

public class ImageProcess {

    private static int dynamicRange = 8, k = 2, rank = 3, power = 5, height, width, stackSize, originalStackSize;
    private static double tau = 7;
    private static MODE mode = MODE.SOFT;
    private enum MODE {SOFT, HARD};
    private static double tol = 0.001;
    private static JPanel previewPanel;
    private static JPanel backgroundImagePanel;
    private static JPanel sparseImagePanel;

//    public ImageProcess(JPanel previewPanel, JPanel backgroundImagePanel, JPanel sparseImagePanel){
//        ImageProcess.previewPanel = previewPanel;
//        ImageProcess.backgroundImagePanel = backgroundImagePanel;
//        ImageProcess.sparseImagePanel = sparseImagePanel;
//    }

    public void setDynamicRange(int type){
        dynamicRange = type;
    }
    public void setK(int k){
        ImageProcess.k = k;
    }
    public void setTau(int tau){
        ImageProcess.tau = tau;
    }
    public void setMode(MODE mode){
        ImageProcess.mode = mode;
    }
    public void setRank(int rank){
        ImageProcess.rank = rank;
    }
    public void setPower(int power){
        ImageProcess.power = power;
    }
    public void setTol(int tol){
        ImageProcess.tol = tol;
    }
    public void setWidth(int width){
        ImageProcess.width = width;
    }
    public void setHeight(int height){
        ImageProcess.height = height;
    }
    public void setStackSize(int stackSize){
        ImageProcess.stackSize = stackSize;
    }
    public void setOriginalStackSize(int stackSize){
        ImageProcess.originalStackSize = stackSize;
    }
    public int getOriginalStackSize(){
        return ImageProcess.originalStackSize;
    }

    public void setPreviewPanel(JPanel previewPanel){
        ImageProcess.previewPanel = previewPanel;
    }

    public void setBackgroundImagePanel(JPanel backgroundImagePanel){
        ImageProcess.backgroundImagePanel = backgroundImagePanel;
    }

    public void setSparseImagePanel(JPanel sparseImagePanel){
        ImageProcess.sparseImagePanel = sparseImagePanel;
    }


    public void process(ImagePlus imp) {


        long startTime = System.nanoTime();

        /** originalImg
         * It is matrix[rows*cols] where:
         *   rows = stackSize
         *   cols = width*height
         * This means that each line represents one of the layers of the tif image
         *
         */
        SimpleMatrix originalImg = constructMatrix(imp.getStack(), dynamicRange);

        //transposing
        originalImg = originalImg.transpose();

        /*
         * SVD decomposition
         * svdResult = ArrayList<X, Y, s>
         * X = Unitary matrix having left singular vectors as columns.
         * Y = Unitary matrix having right singular vectors as rows.
         * s = Diagonal matrix with singular values.
         */
        //ArrayList<SimpleMatrix> svdResult = svdDecomposition(originalImg);
        ArrayList<SimpleMatrix> svdResult = randomizedSVD(originalImg, k);

        SimpleMatrix X = svdResult.get(0);
        SimpleMatrix Y = svdResult.get(1);
        SimpleMatrix s = svdResult.get(2);

        System.gc(); // appel du arbage collector
        Runtime.getRuntime().gc();

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

//        double[][] A2 = matrix2Array(originalImg);
        double[][] L2 = matrix2Array(L);
        double[][] S2 = matrix2Array(S);
        double[][] G2 = matrix2Array(G);

//        ImagePlus original = new ImagePlus(" Original Image ", constructImageStack(
//                A2, dynamicRange));
        ImagePlus im = new ImagePlus(" Background Image ", constructImageStack(L2, dynamicRange));
        ImagePlus im2 = new ImagePlus(" Sparse Image ", constructImageStack(S2, dynamicRange));
        /* Construction du stack d'images pour le bruit (noise) */
        ImagePlus noise = new ImagePlus("Noise Image", constructImageStack(G2, dynamicRange));


        int index = previewPanel.getComponentZOrder(backgroundImagePanel);
        previewPanel.remove(backgroundImagePanel);
        backgroundImagePanel = MyGUI.createPreviewWindow(im);
        previewPanel.add(backgroundImagePanel, index);

        index = previewPanel.getComponentZOrder(sparseImagePanel);
        previewPanel.remove(sparseImagePanel);
        sparseImagePanel = MyGUI.createPreviewWindow(im2);
        previewPanel.add(sparseImagePanel, index);

        previewPanel.revalidate();
        previewPanel.repaint();


//        im.show();
//        im2.show();
//        original.show();
//        noise.show();        // Affichage du bruit (noise)

        long endTime = System.nanoTime();
        long duration = endTime - startTime;
        long durationInMilliseconds = duration / 1000000;
        double durationInSeconds = duration / 1000000000.0;
        System.out.println("Execution time in nanoseconds: " + duration);
        System.out.println("Execution time in milliseconds: " + durationInMilliseconds);
        System.out.println("Execution time in seconds: " + durationInSeconds);
    }

    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static SimpleMatrix constructMatrix(ImageStack stack, int bit) {
        double[][] matrix = new double[stackSize][width * height];

        if (bit == 8) {
            for (int z = 0; z < stackSize; z++) {
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

    public static ArrayList<SimpleMatrix> randomizedSVD(SimpleMatrix in, int k) {
        int n = in.numCols();
        DenseMatrix64F A = in.getMatrix();
        // Etape 1: Generer une matrice aleatoire R de taille n x k
        SimpleMatrix RR = SimpleMatrix.random(n, k, 0, 1, new java.util.Random()); // imperativement aleatoire entre 0
        // et 1

        DenseMatrix64F R = new DenseMatrix64F(n, k);
        for (int x = 0; x < n; x++) {
            for (int z = 0; z < k; z++) {
                R.set(x, z, RR.get(x, z));
            }
        }
        // Etape 2: Calculer le produit matriciel Y = A * R
        DenseMatrix64F Y = new DenseMatrix64F(A.numRows, R.numCols);
        CommonOps.mult(A, R, Y);

        // Etape 3: Effectuer une decomposition QR sur la matrice Y
        QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(Y.numRows, Y.numCols);
        double[][] data = new double[Y.numRows][Y.numCols];
        for (int i = 0; i < Y.numRows; i++) {
            for (int j = 0; j < Y.numCols; j++) {
                data[i][j] = Y.get(i, j);
            }
        }
        qr.decompose(new DenseMatrix64F(data));
        DenseMatrix64F Qs = qr.getQ(null, true);

        // Etape 4: Calculer la matrice B = Q^T * A
        DenseMatrix64F Q = new DenseMatrix64F(Qs.numCols, Qs.numRows);
        CommonOps.transpose(Qs, Q);
        DenseMatrix64F B = new DenseMatrix64F(Q.numRows, A.numCols);
        CommonOps.mult(Q, A, B);

        // Etape 5: Appliquer la SVD sur la matrice B
        SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(B.numRows, B.numCols,
                true, true, true);
        DenseMatrix64F BB = new DenseMatrix64F(B.numRows, B.numCols);
        for (int i = 0; i < B.numRows; i++) {
            for (int j = 0; j < B.numCols; j++) {
                BB.set(i, j, B.get(i, j));
            }
        }
        svd.decompose(BB);

        DenseMatrix64F U_ss = svd.getU(null, false);
        DenseMatrix64F S_ss = svd.getW(null);
        DenseMatrix64F V_ss = svd.getV(null, false);

        // Etape 6: Calculer la matrice U de la decomposition en valeurs singulieres
        // tronquees de la matrice d'entree A
        DenseMatrix64F Us = new DenseMatrix64F(Qs.numRows, U_ss.numCols);
        CommonOps.mult(Qs, U_ss, Us);

        DenseMatrix64F V_s = new DenseMatrix64F(V_ss.numCols, V_ss.numRows);
        CommonOps.transpose(V_ss, V_s);

        ArrayList<SimpleMatrix> result = new ArrayList<>();
        result.add(SimpleMatrix.wrap(Us));
        result.add(SimpleMatrix.wrap(V_s));
        result.add(SimpleMatrix.wrap(S_ss));

        // Renvoyer les matrices U, S et V de la Truncated SVD
        return result;
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
        for (int z = 0; z < stackSize; z++) {
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
}
