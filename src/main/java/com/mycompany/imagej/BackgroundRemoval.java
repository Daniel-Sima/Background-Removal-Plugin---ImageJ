/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.mycompany.imagej;

import ij.ImagePlus;
import net.imglib2.type.numeric.RealType;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.io.File;

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
@Plugin(type = Command.class, menuPath = "Plugins>Background Removal")
public class BackgroundRemoval<T extends RealType<T>> implements Command {

    //
    // Feel free to add more parameters here...
    //
//    @Parameter
//    private Dataset currentData;
//    @Parameter
//    private UIService uiService;
//    @Parameter
//    private OpService opService;
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

    private static MyGUI.PreviewButtonActionListener previewButtonActionListener;
    private static JButton chooseFileButton;
    private static JLabel pathLabel;
    private static JPanel originalImagePanel;
    private static JPanel backgroundImagePanel;
    private static JPanel sparseImagePanel;
    private static JPanel noiseImagePanel;
    private static JPanel previewPanel;
    static Color[] presetColors = { new Color(255,255,255), new Color(192,192,192), new Color(213,170,213), new Color(170,170,255), new Color(170,213,255), new Color(170,213,170),new Color(255,255,170), new Color(250,224,175), new Color(255,170,170) };
    static Color bgColor;
    private static JFrame frame;
    private static ImagePlus imp;
    JButton previewButton;


    public static void main(final String... args) throws Exception {
         execute();
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/
    public static void execute() {
        generateInterface();
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/


    /*-----------------------------------------------------------------------------------------------------------------------*/

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
//        String[] choice = {"soft", "hard"};
//        GenericDialog d = new GenericDialog("Low Rank and Sparse tool");
//        d.addNumericField("tau:", tau, 2);
//        d.addChoice("soft or hard thresholding", choice, choice[0]);
//        d.showDialog();
//        if (d.wasCanceled()) return;
//        // recuperation de la valeur saisie par l'user
//        tau = d.getNextNumber();
//        String filePath = d.getNextString();
//        int c = d.getNextChoiceIndex();
//        if (c == 0) mode = MODE.SOFT;
//        else mode = MODE.HARD;
//        if (d.invalidNumber()) {
//            IJ.error("Invalid parameters");
//            return;
//        }

        // Create a JFrame and add the JScrollPane to it
        frame = MyGUI.getGUIFrame();


        //frame.pack();
        frame.setVisible(true);

        // Loop through each frame in the TIFF stack and repaint the canvas
//        for (int i = 1; i <= numFrames; i++) {
//            imp.setSlice(i);
//            canvas.repaint();
//            try {
//                Thread.sleep(100); // Add a delay between frames for smoother preview
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
    }


    /*-----------------------------------------------------------------------------------------------------------------------*/

}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
