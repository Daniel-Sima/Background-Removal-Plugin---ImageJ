/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.mycompany.imagej;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.io.FileInfo;
import ij.process.ByteProcessor;
import ij.process.ShortProcessor;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.type.numeric.RealType;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
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
    public static ImageProcess importImage(ImagePlus imp, int maxFrames) {

        ImageProcess processor = new ImageProcess();

        int stackSize = imp.getStackSize();
        int ImgHeight = imp.getHeight();
        int ImgWidth = imp.getWidth();
        int dynamicRange = 0;
        int nbSlices = stackSize;

        // FIXME probleme s'il y a qu'une seule frame ?
        if (imp.getStackSize() < 2) {
            IJ.error("Stack required");
            throw new RuntimeException("Stack required");
        } else {
            nbSlices = maxFrames == 0 ? stackSize : maxFrames;
            if (imp.getType() == ImagePlus.GRAY8) {
                dynamicRange = 8;
            } else if (imp.getType() == ImagePlus.GRAY16) dynamicRange = 16;
            else {
                IJ.error("Image type not supported ( only GRAY8 and GRAY16 )");
            }
        }

        processor.setDynamicRange(dynamicRange);
        processor.setWidth(ImgWidth);
        processor.setHeight(ImgHeight);
        processor.setStackSize(nbSlices);
        processor.setOriginalStackSize(stackSize);

        return processor;
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
        frame = new JFrame("TIFF Stack Preview");
        frame.setSize(1100, 750);
        frame.setLocationRelativeTo(null); // Center frame on screen
        //frame.setLayout(new FlowLayout());

        JPanel mainLayout = new JPanel();
        mainLayout.setPreferredSize(frame.getPreferredSize());

        JPanel loadFilePanel = createFileButton();
        mainLayout.add(loadFilePanel);

        previewPanel = getRowPanel();

        // Load the spinner loading GIF
        ImageIcon loadingIcon = new ImageIcon(GaussFiltering.class.getResource("/icons/loading.gif"));

        // Create the label with the icon
        JLabel loadingLabel = new JLabel(loadingIcon);
        JLabel loadingLabel2 = new JLabel(loadingIcon);

        JButton previewButton = new JButton("Preview");
        previewButton.setPreferredSize(new Dimension(200, 30));

        previewButtonActionListener =
                new MyGUI.PreviewButtonActionListener(previewPanel, backgroundImagePanel, sparseImagePanel, loadingLabel, loadingLabel2);

        chooseFileButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Handle choose file button click
                FileDialog fd = new FileDialog(frame, "Choose Image", FileDialog.LOAD);
                fd.setVisible(true);
                String path = fd.getDirectory() + fd.getFile();
                //pathLabel.setMaximumSize(new Dimension(200, pathLabel.getPreferredSize().height));
                pathLabel.setText(fd.getFile()); // Update path label text

                //TODO:check extension
                imp = IJ.openImage(path);

                ImageProcess processor = importImage(imp, 0);

                int index = previewPanel.getComponentZOrder(originalImagePanel);
                previewPanel.remove(originalImagePanel);
                originalImagePanel = MyGUI.createPreviewWindow(imp);
                previewPanel.add(originalImagePanel, index);

                previewPanel.revalidate();
                previewPanel.repaint();

                processor.setPreviewPanel(previewPanel);
                processor.setBackgroundImagePanel(backgroundImagePanel);
                processor.setSparseImagePanel(sparseImagePanel);

                previewButtonActionListener.setImp(imp);
                previewButtonActionListener.setImageProcessor(processor);
            }
        });


        previewButton.addActionListener(previewButtonActionListener);

        mainLayout.add(previewPanel);
        mainLayout.add(previewButton);

        frame.setLayout(new BoxLayout(frame.getContentPane(), BoxLayout.Y_AXIS));
        frame.add(mainLayout, BorderLayout.CENTER);
        //frame.add(previewPanel, BorderLayout.SOUTH);

        bgColor=presetColors[0];
        frame.setBackground(bgColor);

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

    protected static JPanel getRowPanel()
    {
        JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

        originalImagePanel = new JPanel();
        originalImagePanel.setPreferredSize(new Dimension(300, 300));
        originalImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

        backgroundImagePanel = new JPanel();
        backgroundImagePanel.setPreferredSize(new Dimension(300, 300));
        backgroundImagePanel.setLayout(new BorderLayout());
        backgroundImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

        sparseImagePanel = new JPanel();
        sparseImagePanel.setPreferredSize(new Dimension(300, 300));
        sparseImagePanel.setLayout(new BorderLayout());
        sparseImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

        rowPanel.add(originalImagePanel);
        rowPanel.add(Box.createHorizontalStrut(50));
        rowPanel.add(backgroundImagePanel);
        rowPanel.add(Box.createHorizontalStrut(50));
        rowPanel.add(sparseImagePanel);

        return rowPanel;
    }

    protected static JPanel createFileButton()
    {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));
        panel.setPreferredSize(new Dimension(frame.getWidth()-200, 30));

        JPanel buttonPanel = new JPanel();
        buttonPanel.setPreferredSize(new Dimension((int)(0.2 * panel.getPreferredSize().getWidth()), (int) panel.getPreferredSize().getHeight()));
        buttonPanel.setLayout(new BorderLayout());

        chooseFileButton = new JButton("Choose File");
        chooseFileButton.setPreferredSize(new Dimension(400, 30));

        buttonPanel.add(chooseFileButton, BorderLayout.CENTER);

        pathLabel = new JLabel("No file selected");
        pathLabel.setPreferredSize(new Dimension((int)(0.5 * panel.getPreferredSize().getWidth()), pathLabel.getPreferredSize().height));
        pathLabel.setHorizontalAlignment(JLabel.LEFT);

        JPanel labelPanel = new JPanel();
        labelPanel.setPreferredSize(new Dimension((int)(0.5 * panel.getPreferredSize().getWidth()), (int) panel.getPreferredSize().getHeight()));
        labelPanel.setLayout(new BorderLayout());
        labelPanel.add(pathLabel, BorderLayout.CENTER);

        panel.add(buttonPanel);
        panel.add(Box.createHorizontalStrut(50));
        panel.add(labelPanel);

        return panel;
    }
    /*-----------------------------------------------------------------------------------------------------------------------*/

}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
