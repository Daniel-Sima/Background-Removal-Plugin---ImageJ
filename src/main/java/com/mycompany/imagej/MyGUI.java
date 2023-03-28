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
import java.util.Objects;

public class MyGUI {

    private static JFrame frame;
    private static JPanel previewPanel;
    private static JPanel originalImagePanel;
    private static JPanel backgroundImagePanel;
    private static JLabel loadingBackgroundLabel;
    private static JLabel loadingSparseLabel;
    private static JLabel loadingNoiseLabel;
    private static JPanel sparseImagePanel;
    private static JButton chooseFileButton;
    private static JLabel pathLabel;
    private static ImagePlus imp;
    private static ImageProcess processor;
    private static Color[] presetColors = { new Color(255,255,255), new Color(192,192,192), new Color(213,170,213), new Color(170,170,255), new Color(170,213,255), new Color(170,213,170),new Color(255,255,170), new Color(250,224,175), new Color(255,170,170) };
    private static Color bgColor;
    private static int nbSlicesPreview;

    public static void importImage(ImagePlus imp, int maxFrames) {

        int stackSize = imp.getStackSize();
        int ImgHeight = imp.getHeight();
        int ImgWidth = imp.getWidth();
        int dynamicRange = 0;
        int nbSlices;

        // FIXME probleme s'il y a qu'une seule frame ?
        if (imp.getStackSize() < 2) {
            IJ.error("Stack required");
            throw new RuntimeException("Stack required");
        } else {
            nbSlices = maxFrames == 0 ? stackSize : maxFrames;
            nbSlicesPreview = Math.min(stackSize, 100);
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

    }
    protected static JFrame getGUIFrame()
    {
        processor = new ImageProcess();
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
        processor.setPreviewPanel(previewPanel);
        processor.setBackgroundImagePanel(backgroundImagePanel);
        processor.setSparseImagePanel(sparseImagePanel);

        // Load the spinner loading GIF
        ImageIcon loadingIcon = new ImageIcon(Objects.requireNonNull(GaussFiltering.class.getResource("/icons/loading.gif")));

        // Create the label with the icon
        loadingBackgroundLabel = new JLabel(loadingIcon);
        loadingSparseLabel = new JLabel(loadingIcon);

        JButton previewButton = new JButton("Preview");
        previewButton.setPreferredSize(new Dimension(200, 30));

        PreviewButtonActionListener previewButtonActionListener = new PreviewButtonActionListener(previewPanel, backgroundImagePanel);

        //processor = new ImageProcess(previewPanel, backgroundImagePanel, sparseImagePanel);
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

                importImage(imp, 0);

                int index = previewPanel.getComponentZOrder(originalImagePanel);
                previewPanel.remove(originalImagePanel);
                originalImagePanel = MyGUI.createPreviewWindow(imp);
                previewPanel.add(originalImagePanel, index);

                previewPanel.revalidate();
                previewPanel.repaint();

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

        return frame;
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

    public static class PreviewButtonActionListener implements ActionListener {
        private JPanel backgroundImagePanel;
        private final JPanel previewPanel;
        private ImagePlus imp;
        private ImageProcess processor;

        public PreviewButtonActionListener(JPanel previewPanel, JPanel backgroundImagePanel) {
            this.backgroundImagePanel = backgroundImagePanel;
            this.previewPanel = previewPanel;
//            this.processor = new ImageProcess();
        }

        public void setImp(ImagePlus imp)
        {
            this.imp = imp;
        }

        public void setImageProcessor(ImageProcess processor)
        {
            this.processor = processor;
        }

        public void actionPerformed(ActionEvent e) {

            backgroundImagePanel.removeAll();
            backgroundImagePanel.add(loadingBackgroundLabel, BorderLayout.CENTER);
            backgroundImagePanel.revalidate();
            backgroundImagePanel.repaint();

            sparseImagePanel.removeAll();
            sparseImagePanel.add(loadingSparseLabel, BorderLayout.CENTER);
            sparseImagePanel.revalidate();
            sparseImagePanel.repaint();

            previewPanel.revalidate();
            previewPanel.repaint();

            SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
                @Override
                protected Void doInBackground() throws Exception {
                    processor.setStackSize(nbSlicesPreview);
                    processor.process(imp);
                    return null;
                }
            };
            worker.execute();
        }
    }
}
