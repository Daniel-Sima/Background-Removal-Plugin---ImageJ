/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.mycompany.imagej;

import net.imagej.Dataset;
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

import Jama.Matrix;

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
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

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
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SingularOps;
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

    @Parameter
    private Dataset currentData;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;

    @Override
    public void run() {
        final Img<T> image = (Img<T>)currentData.getImgPlus();

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
    private static MODE mode = MODE . SOFT ;
    private enum MODE { SOFT , HARD ; }
    private static int height, width, nbSlices, type ;
  
    public static void main(final String... args) throws Exception {
        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ(); 
        
        // affiche l'interface utilisateur pour acceder aux functionnalites 
        ij.ui().showUI();
        
        // ask the user for a file to open
//        final File file = ij.ui().chooseFile(null, "open");
        final File file = new File("C:/Users/skullriver/Desktop/Sorbonne/S2/pstl/Codes/LowRankSparseNoiseDecomposition-GoDec/Samples/1-Original.tif");
        
        // si le fichier choisi est bon
        if (file != null) {
        	// verification qu'il s'agit d'un .tif ou .tiff ou .avi
        	importImage(file, 0);  
        	
        	// recuperation des donnees
        	ImagePlus imp = IJ.openImage(file.getPath());  // image
        	ImageProcessor ip = imp.getProcessor();
        	int autotr = ip.getAutoThreshold();

        	int stackSize = imp.getStackSize();	// nombre de frames
            int ImgHeight = imp.getHeight();	// hauteur
            int ImgWidth = imp.getWidth();	// largeur
                    
            // FIXME probleme s'il y a qu'une seule frame ?
            if ( imp.getStackSize() < 2) {
            	IJ.error (" Stack required ");
            	return;
            } else { // si plusieurs frames
            	nbSlices = stackSize; 
            	if( imp.getType() == ImagePlus.GRAY8 ) type = 8;
            	else 
            		if( imp.getType () == ImagePlus.GRAY16 ) type = 16;
            		else {
            			IJ.error (" Image type not supported ( only GRAY8 and GRAY16 )") ;
            			return ;
            		}
            }
            
            height = ImgHeight;
            width = ImgWidth;
        	
        	String [] choice = {" soft "," hard "};
        	GenericDialog d = new GenericDialog(" Threshold ") ;
        	d.addNumericField("tau :", tau, 2) ;
        	d.addChoice(" soft or hard thresholding ", choice , choice [0]) ;
        	d.showDialog() ;
        	if( d.wasCanceled() ) return ;
        	
        	// recuperation de la valeur saisie par l'user
        	tau = d.getNextNumber() ; 
        	int c = d.getNextChoiceIndex() ;
        	if( c == 0) mode = MODE.SOFT ;
        	else mode = MODE.HARD ;
        	if( d.invalidNumber() ) {
        		IJ.error(" Invalid parameters ") ;
        		return ;
        	}
            
            double[][][] matrix = constructMatrix(imp.getStack(), type) ;
            
            /*
             * SVD 
             */
            
            int nb = 0; // 12h30
            DenseMatrix64F A = new DenseMatrix64F(nbSlices, height*width);
            for (int i=0; i<nbSlices; i++) {
            	for (int j=0; j<height; j++) {
            		for (int e=0; e<width; e++) {
            			A.set(i, nb, matrix[i][e][j]);
            			nb++;
            		}
            	}
            	nb = 0;
            }

//            ok = ok.transpose();

            DenseMatrix64F ok = new DenseMatrix64F(A.numCols, A.numRows);

            CommonOps.transpose(A, ok);

            //old svd

            /*SimpleMatrix U=null,W=null,V=null, check = null;
            @SuppressWarnings("unchecked")
			SimpleSVD<SimpleMatrix> svd = ok.svd(true); // compacte pour aller plus vite ...

            U=svd.getU();
            W=svd.getW();
            V=svd.getV();

            SimpleMatrix X = svd.getU().extractMatrix(0, U.numRows(), 0, k); // CMMT
            SimpleMatrix invX = new SimpleMatrix(X.numRows(), X.numCols());
            for (int e=X.numCols()-1, i=0; i<X.numCols(); i++, e--) {
            	for (int j=0; j<X.numRows(); j++) {
            		invX.set(j, e, X.get(j, i));
            	}
            }*/

            org.ejml.factory.SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(ok.numRows, k, true, true, true);

            svd.decompose(ok);
            DenseMatrix64F U = svd.getU(null, false);
            DenseMatrix64F W = svd.getW(null);
            DenseMatrix64F V = svd.getV(null, false);

            DenseMatrix64F X = new DenseMatrix64F(U.numRows, k);
            CommonOps.extract(U, 0, U.numRows, 0, k, X, 0, 0);

            DenseMatrix64F invX = new DenseMatrix64F(X.numRows, X.numCols);
            for (int e=X.numCols-1, i=0; i<X.numCols; i++, e--) {
                for (int j=0; j<X.numRows; j++) {
                    invX.set(j, e, X.get(j, i));
                }
            }

            // mult par -1 pour changer de signe pour avoir les mêmes val qu'en python psq jsp pq y avait un -
            SimpleMatrix negatif = SimpleMatrix.diag(1, -1);
//            invX = invX.mult(negatif);
//
//            SimpleMatrix Y = svd.getV().extractMatrix(0, k, 0, V.numCols()); // CMMT
//            SimpleMatrix Ytest = svd.getV().extractMatrix(0, V.numCols(), 0, k); // CMMT
//            SimpleMatrix invY = new SimpleMatrix(Ytest.numRows(), Ytest.numCols());
//            for (int e=Ytest.numCols()-1, i=0; i<Ytest.numCols(); i++, e--) {
//            	for (int j=0; j<Ytest.numRows(); j++) {
//            		invY.set(j, e, Ytest.get(j, i));
//            	}
//            }
//
//            invY = invY.mult(negatif);
//
//            invY = invY.transpose();
//
//            SimpleMatrix s = svd.getW().extractMatrix(0, k, 0, k);
//            SimpleMatrix ss = s.extractDiag();
//
//            double[] valS = new double[ss.numRows()];
//            for (int j=0, i=ss.numRows()-1; i>=0; i--, j++) {
//            	valS[j] = ss.get(i, 0);
//            }
//
//            SimpleMatrix ssi = SimpleMatrix.diag(valS);
//
//            X = invX.mult(ssi);
//
//            Y = invY;
//            SimpleMatrix L = X.mult(Y);
//
//            SimpleMatrix S = ok.minus(L);
//
//            S = threshold(S, tau, "soft");
//
//            double result = (double) rank / k;
//            int rankk = (int) Math.round(result);
//            SimpleMatrix error = new SimpleMatrix(rank * power, 1);
//            SimpleMatrix T = (ok.minus(L)).minus(S);
//            double normD = ok.normF();
//            double normT = T.normF();
//
//            error.set(0, (double) normT / normD);
//
//            int iii = 1;
//            boolean stop = false;
//            double alf = 0;
//
//            for(int i=1; i<rankk+1; i++) {
//            	i = i-1;
//            	int rrank = rank;
//            	int est_rank = 1;
//            	alf = 0;
//            	double increment = 1;
//
//            	if (iii == power * (i - 2) + 1) {
//            		iii = iii + power;
//            	}
//
//            	for(int j=1; j<power+1; j++) {
//
//            		//update X
//            		X = (L.mult(Y.transpose()));
//            		for (int k=0; k<X.numCols(); k++) {
//            			for (int l=0; l<X.numRows(); l++) {
//            				X.set(l, k, Math.abs(X.get(l, k)));
//            			}
//            		}
//
//                    DenseMatrix64F X2 = X.getMatrix();
//
//            		// Do a QR decomposition
//            		/* Possible qu'ici la decomposition se fasse mal */
//
//                    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X2.numRows, X2.numCols);
//                    qr.decompose(X2);
//                    X = SimpleMatrix.wrap(qr.getQ(null, true));
//
//                    if (j==1 && i==1) {
//                        X.saveToFileCSV("Y.csv");
//                    }
//
//
//                    // Update of Y
//                    Y = (X.transpose()).mult(L);
//                    L = X.mult(Y);
//
//                    // Update of S
//                    T = ok.minus(L);
//                    //thresholding
//                    S = threshold(T, tau, "soft");
//
//
//                    // Error, stopping criteria
//                    T = T.minus(S);
//                    int ii = iii + j - 1;
//
//                    normT = T.normF();
//
//                    error.set(ii, (double) normT / normD);
//
//                    if (error.get(ii) < tol) {
//                    	stop = true;
//                    	System.out.println("STOPPPPPPPPPP");
//                    	break;
//                    }
//
//                    if (rrank != rank) {
//                    	System.out.println("AICI BA");
//                    	rank = rrank;
//                    }
//
//                    // Adjust alf
//                    double ratio = (double) error.get(ii) / error.get(ii-1);
//
//                    // Intermediate variables
//                    SimpleMatrix X1 = X;
//                    SimpleMatrix Y1 = Y;
//                    SimpleMatrix L1 = L;
//					SimpleMatrix S1 = S;
//                    SimpleMatrix T1 = T;
//
//
//                    if (ratio >= 1.1) {
////                    	System.out.println("1er");
//                    	increment = Math.max(0.1 * alf, 0.1 * increment);
//                        error.set(ii, error.get(ii-1));
//                        alf = 0;
//                    }else if (ratio > 0.7) {
////                    	System.out.println("2eme");
//                    	increment = Math.max(increment, 0.25 * alf);
//                        alf = alf + increment;
//                    }
//
//                    // Update of L
//                    L = L.plus(T.scale(1+alf));
//
//                    // Add corest AR
//                    if (j > 8) {
//                    	System.out.println("PULA");
//                        double mean = mean(error, ii - 7, ii);
//                        if (mean > 0.92) {
//                            iii = ii;
//                            int YRow = Y.numRows();
//                            int XRow = X.numRows();
//                            if ((YRow - XRow) >= k) {
//                                Y = Y.extractMatrix(0, XRow - 1, 0, Y.numCols());
//                            }
//                            break;
//                        }
//                    }
//
//
//            	}
//
//            	if(stop) {
//            		break;
//            	}
//
////            	AR
//            	if (i + 1 < rankk) {
//            		Random r = new Random();
//            	    SimpleMatrix RR = new SimpleMatrix(k, ok.numRows());
//            	    for (int x=0; x<k; x++) {
//            	    	for (int z=0; z<ok.numRows(); z++) {
////            	    		RR.set(x, z, r.nextGaussian());
//            	    		RR.set(x, z, 1.0);
//            	    	}
//            	    }
//
//            	    SimpleMatrix v = RR.mult(L);
//            	    Y = Y.combine(2, 0, v);
//            	}
//
//            	i++;
//
//            }
//            L = X.mult(Y);
//
//
//            System.out.println(L.numRows()+" "+L.numCols());
//            if (ok.numRows() > ok.numCols()){
//            	L = L.transpose();
//                assert S != null;
//                S = S.transpose();
//            	ok = ok.transpose();
//            }
//
//            double[][] A2 = matrix2Array2(ok);
//            double[][] L2 = matrix2Array2(L);
//            assert S != null;
//            double[][] S2 = matrix2Array2(S);
//
//
//            ImagePlus original = new ImagePlus (" Original Image ", constructImageStack2 (
//            		A2, type) ) ;
//            ImagePlus im = new ImagePlus (" Background Image ", constructImageStack2 (
//            		L2, type) ) ;
//            ImagePlus im2 = new ImagePlus (" Sparse Image ", constructImageStack2 (
//            		S2, type) ) ;
//
//
//            im.show();
//            im2.show();
//            original.show();
            
            /*
             * Maybe will be useful
             */
            
            // load the dataset
//            final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());
//            
//            Img img = dataset.getImgPlus().getImg();
//            
//            long[] dims = new long[img.numDimensions()];
//            img.dimensions(dims);
//            byte[] data = ArrayImgs.unsignedBytes(dims).update(img).getCurrentStorageArray();
//            

            /*
             * END: Maybe will be useful
             */
            
            // show the image
            //ij.ui().show(dataset);

            // invoke the plugin
            //ij.command().run(GaussFiltering.class, true);
        }
    }
    
/*-----------------------------------------------------------------------------------------------------------------------*/
    public static void importImage(File file, int maxFrames) {
        String fileExtension = getFileExtension(file);

        if (fileExtension.equals("tif") || fileExtension.equals("tiff")) {
            System.out.println("TIF stack loading OK");
        } else if (fileExtension.equals("avi")) {
            // Code to load AVI file
            System.out.println("AVI loading OK");
        } else {
            throw new RuntimeException("The file extension should be .tif, .tiff or .avi.");
        }
    }
/*-----------------------------------------------------------------------------------------------------------------------*/
    public static String getFileExtension(File file) {
        String fileName = file.getName();
        if(fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0)
        return fileName.substring(fileName.lastIndexOf(".")+1);
        else return "";
    }
/*-----------------------------------------------------------------------------------------------------------------------*/   
    /*
     * Code from last year's project
     */
    private static double[][][] constructMatrix ( ImageStack stack , int bit ) {
    	System.out.println(nbSlices);
        System.out.println(height);
        System.out.println(width);
    	double [][][] matrix = new double[nbSlices][width][height];
    	if( bit == 8) {
    		for (int z = 0; z < nbSlices ; z++) {
    			ByteProcessor bp = ( ByteProcessor ) stack.getProcessor (z+1) ;
    			for (int i = 0; i < width ; i ++)
    				for (int j =0; j < height ; j ++)
    					matrix[z][i][j] = bp.getPixelValue(i, j ) ;
    		}
    	}
    	if( bit == 16) {
    		for (int z = 0; z < nbSlices ; z ++) {
    			ShortProcessor bp = ( ShortProcessor ) stack.getProcessor (z+1) ;
    			for (int i = 0; i < width ; i ++)
    				for (int j =0; j < height ; j ++)
    					matrix[z][i][j] = bp.getPixelValue (i, j) ;
    	
    		}
    	}
    	return matrix ;
    }
/*-----------------------------------------------------------------------------------------------------------------------*/
    private static ImageStack constructImageStack ( double [][][] matrix , int bit ) {
    	ImageStack newStack = new ImageStack ( width , height ) ;
        for (int z = 0; z < nbSlices ; z ++) {
            ByteProcessor bp = new ByteProcessor(width, height);

            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    bp.putPixel(i, j, (int) matrix[z][i][j]);
            newStack.addSlice(bp);
        }
    	
    	return newStack;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    private static ImageStack constructImageStack2 ( double [][] matrix , int bit ) {    	
    	ImageStack newStack = new ImageStack (width, height) ;
        for (int z = 0; z < nbSlices ; z ++) {
        	byte[] pixels = new byte[width*height];
        	for (int i = 0; i < pixels.length; i++) {
        	    pixels[i] = (byte) matrix[z][i];
        	}
            ByteProcessor bp = new ByteProcessor(width, height, pixels);
                    
            newStack.addSlice(bp);
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
        if (mode.equals("soft")) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data.get(i, j);
                    if (Math.abs(val) < tau) {
                    	result.set(i, j, 0);
                    } else {
                    	result.set(i, j, (val/Math.abs(val) * Math.max(Math.abs(val) - tau, 0)));
                    }
                }
            }
        }
        else if (mode.equals("hard")) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data.get(i, j);
                    if (Math.abs(val) < tau) {
                        result.set(i, j, 0);
                    }
                    else {
                    	result.set(i, j, val);
                    }
                }
            }
        }
        else {
            System.out.println("mode not supported");
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
	                int row = (c % (width*height)) / height;
	                int col = (c % (width*height)) % height;
	                array[r][row][col] = matrix.get(r, c);
	            }
	        }
	        return array;
	    }
	 
	 private static double[][][] matrix2Array3(SimpleMatrix matrix) {
	        double[][][] array = new double[nbSlices][width][height];
	        for (int r = 0; r < matrix.numRows(); r++) {
	            for (int c = 0; c < matrix.numCols(); c++) {
	                int row = (c % (width*height)) / height;
	                int col = (c % (width*height)) % height;
	                array[r][row][col] = matrix.get(r, c);
	            }
	        }
	        return array;
	    }
	 	
}
