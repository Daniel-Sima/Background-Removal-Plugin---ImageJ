package com.mycompany.imagej;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileInfo;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;

public class MyGUI {


    protected static JPanel createPreviewWindow(ImagePlus imp)
    {
        if(imp != null){
            // Get the number of frames in the TIFF stack
            int numFrames = imp.getStackSize();

            // Create a custom Canvas to display the preview
            StackCanvas canvas = new StackCanvas(imp);

            // Create a panel to hold the canvas with a fixed size
            JPanel placeholder = new JPanel();
            placeholder.setPreferredSize(new Dimension(300, 300));

            // Create a JSlider to slide frames manually
            JSlider slider = new JSlider(1, numFrames, 1);
            slider.setPreferredSize(new Dimension((int)placeholder.getPreferredSize().getWidth(), slider.getPreferredSize().height));

            slider.addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent e) {
                    int slice = slider.getValue();
                    imp.setSlice(slice);
                    canvas.repaint();
                }
            });

            // Create the first row JPanel and set its preferred size to take up 90% of the space
            JPanel row1 = new JPanel();
            row1.setPreferredSize(new Dimension((int)placeholder.getPreferredSize().getWidth(), (int)(0.9 * placeholder.getPreferredSize().getHeight())));
            row1.setLayout(new BorderLayout());
            // Add the canvas to the center of the first row panel
            row1.add(canvas,  BorderLayout.CENTER);

            // Create the second row JPanel and set its preferred size to take up 10% of the space
            JPanel row2 = new JPanel();
            //row2.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
            row2.setPreferredSize(new Dimension((int)placeholder.getPreferredSize().getWidth(), (int)(0.1 * placeholder.getPreferredSize().getHeight())));
            row2.setLayout(new BorderLayout());
            // Add the slider to the center of the second row panel
            row2.add(slider, BorderLayout.CENTER);

            // Add the first and second row panels to the main panel
            placeholder.setLayout(new BoxLayout(placeholder, BoxLayout.PAGE_AXIS));
            placeholder.add(row1);
            placeholder.add(row2);


            return placeholder;
        }
        return null;
    }

    private static class StackCanvas extends Canvas {
        private static final long serialVersionUID = 1L;

        private final ImagePlus imp;
        private final int imageWidth;
        private final int imageHeight;

        public StackCanvas(ImagePlus imp) {
            this.imp = imp;
            this.imageWidth = imp.getWidth();
            this.imageHeight = imp.getHeight();
//            setPreferredSize(new Dimension(imageWidth, imageHeight));
        }

        @Override
        public void paint(Graphics g) {

            // Create off-screen buffer
            Image offscreen = createImage(getWidth(), getHeight());
            Graphics buffer = offscreen.getGraphics();


            // Calculate the scale factor for the image
            double scale = Math.min((double)getWidth() / imp.getWidth(), (double)getHeight() / imp.getHeight());

            if (imp.getWidth() < getWidth() && imp.getHeight() < getHeight()){
                scale = 1.0;
            }

            // Get the current slice and the corresponding image processor
            int slice = imp.getCurrentSlice();
            FileInfo fi = imp.getFileInfo();
            BufferedImage img = imp.getStack().getProcessor(slice).getBufferedImage();

            int scaledWidth = (int) (imp.getWidth()*scale);
            int scaledHeight = (int)(imp.getHeight()*scale);

            // Calculate the offset needed to center the image on the canvas
            int offsetX = (getWidth() - scaledWidth) / 2;
            int offsetY = (getHeight() - scaledHeight) / 2;

            // Draw the image on the off-screen buffer
            buffer.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight, null);


            //TODO:replace this sliceText

            // Draw the slice number and file name in the bottom-left corner
            String sliceText = "Slice: " + slice + "/" + imp.getStackSize();
            buffer.drawString(sliceText, 10, getHeight() - 10);

            // Swap the buffers
            g.drawImage(offscreen, 0, 0, getWidth(), getHeight(),null);
        }
    }
}
