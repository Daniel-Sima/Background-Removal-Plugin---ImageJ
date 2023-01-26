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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

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

import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DenseMatrix64F;
import org.ejml.dense.row.SingularOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.ops.CommonOps;
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
    private static double tau = 230;
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
        ij.ui().showUI();

        // ask the user for a file to open
        final File file = ij.ui().chooseFile(null, "open");

        if (file != null) {
        	
        	importImage(file, 0);
        	
        	ImagePlus imp = IJ.openImage(file.getPath());
        	int stackSize = imp.getStackSize();
            int ImgHeight = imp.getHeight();
            int ImgWidth = imp.getWidth();
                      
            if ( imp.getStackSize() < 2) {
            	IJ.error (" Stack required ");
            	return;
            } else {
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
            
            System.out.println(stackSize);
            System.out.println(ImgHeight);
            System.out.println(ImgWidth);
        	
        	String [] choice = {" soft "," hard "};
        	GenericDialog d = new GenericDialog(" Threshold ") ;
        	d.addNumericField("tau :", tau, 2) ;
        	d.addChoice(" soft or hard thresholding ", choice , choice [0]) ;
        	d.showDialog() ;
        	if( d.wasCanceled() ) return ;
        	
        	tau = d.getNextNumber() ;
        	int c = d.getNextChoiceIndex() ;
        	if( c == 0) mode = MODE.SOFT ;
        	else mode = MODE.HARD ;
        	if( d.invalidNumber() ) {
        		IJ.error(" Invalid parameters ") ;
        		return ;
        	}
        	
        	System.out.println(tau);
            System.out.println(mode);
            
            double[][][] matrix = constructMatrix(imp.getStack(), type) ;
            
            /*
             * SVD 
             */
            SimpleMatrix A = new SimpleMatrix(matrix[0]);
                     
            SimpleSVD<SimpleMatrix> svd = A.svd();

            SimpleMatrix U=null,W=null,V=null, check = null;
            
			double [] Svd = null;
            Svd = svd.getSVD().getSingularValues();
            U=svd.getU();
            W=svd.getW();
            V=svd.getV(); 
            SimpleMatrix X = svd.getU().extractMatrix(0, U.numRows(), 0, k);
            SimpleMatrix Y = svd.getV().extractMatrix(0, k, 0, V.numCols());
            SimpleMatrix s = svd.getW().extractMatrix(0, k, 0, k);
            
            X = X.mult(s);
            SimpleMatrix L = X.mult(Y);
            
            SimpleMatrix S = A.minus(L);
            
            //thresholding
            S = thresholding(S);        
            
            double result = (double) rank / k;
            int rankk = (int) Math.round(result);
            SimpleMatrix error = new SimpleMatrix(rank * power, 1);
            SimpleMatrix T = (A.minus(L)).minus(S);
            double normD = A.normF();
            double normT = T.normF();
            
            error.set(0, (double) normT / normD);
            
            int iii = 1;
            boolean stop = false;
            double alf = 0;
            
            for(int i=0; i<rankk; i++) {
            	int rrank = rank;
            	int est_rank = 1;
            	alf = 0;
            	double increment = 1;

            	if (iii == power * (i - 2) + 1) {
            		iii = iii + power;
            	}
            	
            	for(int j=1; j<power+1; j++) {
            		
            		//update X 
            		//TODO: absolute values 
            		X = (L.mult(Y.transpose()));
            		
            		DenseMatrix64F X2 = X.getMatrix();
            		
            		// Do a QR decomposition
                    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X.numRows(), X.numCols());
                    qr.decompose(X2);
                    DenseMatrix64F qm = qr.getQ(null, true);
                    DenseMatrix64F rm = qr.getR(null, true);
                    
                    X = SimpleMatrix.wrap(qm);
            
                    // Update of Y
                    Y = (X.transpose()).mult(L);
                    L = X.mult(Y);
                    
                    // Update of S
                    T = A.minus(L);
                    //thresholding
                    S = thresholding(T);
                    
                    // Error, stopping criteria
                    T = T.minus(S);
                    int ii = iii + j - 1;
                    
                    normT = T.normF();
                    
                    error.set(ii, (double) normT / normD);
                    
                    if (error.get(ii) < tol) {
                    	stop = true;
                    	break;
                    }
                        
                    if (rrank != rank) {
                    	rank = rrank;
                    	if(est_rank == 0) {
                    		alf = 0;
                    		continue;
                    	}
                    }
                    
                    // Adjust alf
                    double ratio = (double) error.get(ii) / error.get(ii-1);
                    
                    // Intermediate variables
                    SimpleMatrix X1 = X;
                    SimpleMatrix Y1 = Y;
                    SimpleMatrix L1 = L;
					SimpleMatrix S1 = S;
                    SimpleMatrix T1 = T;
                    
                    
                    if (ratio >= 1.1) {
                    	increment = Math.max(0.1 * alf, 0.1 * increment);
                    	X = X1;
                        Y = Y1;
                        L = L1;
                        S = S1;
                        T = T1;
                        error.set(ii, error.get(ii-1));
                        alf = 0;
                    }else if (ratio > 0.7) {
                    	increment = Math.max(increment, 0.25 * alf);
                        alf = alf + increment;
                    }
                    
                    // Update of L
                    X1 = X;
                    Y1 = Y;
                    L1 = L;
                    S1 = S;
                    T1 = T;
                    L = L.plus(T.scale(1+alf));
  
            	}
            	
            	if( stop == true) {
            		break;
            	}

            }
            L = X.mult(Y);
           
            
            double[][] L2 = matrix2Array(L);
            double[][] S2 = matrix2Array(S);
        	
            ImagePlus im = new ImagePlus (" Background Image ", constructImageStack (
            		L2, type ) ) ;
            ImagePlus im2 = new ImagePlus (" Sparse Image ", constructImageStack (
            		S2, type ) ) ;
            
            im.show();
            im2.show();
            
            /*
             * Maybe will be useful
             */
            
            // load the dataset
            final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());
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
    
    public static String getFileExtension(File file) {
        String fileName = file.getName();
        if(fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0)
        return fileName.substring(fileName.lastIndexOf(".")+1);
        else return "";
    }
    
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
    
    private static ImageStack constructImageStack ( double [][] matrix , int bit ) {
    	ImageStack newStack = new ImageStack ( width , height ) ;
    	ByteProcessor bp = new ByteProcessor ( width , height ) ;
    	for ( int i = 0; i < width ; i ++)
    		for ( int j = 0; j < height ; j ++)
    			bp.putPixel (i , j , ( int ) matrix[ i ][ j ]) ;
    	newStack . addSlice ( bp ) ;
    	
    	return newStack;
    }
    private static double[][] matrix2Array(SimpleMatrix matrix) {
        double[][] array = new double[matrix.numRows()][matrix.numCols()];
        for (int r = 0; r < matrix.numRows(); r++) {
            for (int c = 0; c < matrix.numCols(); c++) {
                array[r][c] = matrix.get(r, c);
            }
        }
        return array;
    }
    
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
   

    
    
    

}
