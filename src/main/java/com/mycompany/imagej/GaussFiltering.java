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
import ij.io.Opener;
import ij.process.ByteProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
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
             * SVD but it's not what we want
             */
            SimpleMatrix A = new SimpleMatrix(matrix[0]);
            
            SimpleSVD<SimpleMatrix> s = A.svd();
            SimpleMatrix U=null,W=null,V=null;
            U=s.getU();
            W=s.getW();
            V=s.getV();
        	
            
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

}
