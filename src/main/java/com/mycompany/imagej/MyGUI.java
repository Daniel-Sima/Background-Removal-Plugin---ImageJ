package com.mycompany.imagej;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileInfo;

import javax.swing.*;
import javax.swing.border.BevelBorder;
import javax.swing.border.EtchedBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.util.Objects;

/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/**
 * Class with user interface methods.
 *
 */
public class MyGUI {

	private static JFrame frame;
	private static JPanel previewPanel;
	private static JPanel originalImagePanel;
	private static JPanel backgroundImagePanel;
	private static JLabel loadingBackgroundLabel;
	private static JLabel loadingSparseLabel;
	private static JLabel loadingNoiseLabel;
	private static JPanel sparseImagePanel;
	private static JPanel noiseImagePanel;
	private static JButton chooseFileButton;
	private static JLabel pathLabel;
	private static ImagePlus imp;
	private static ImageProcess processor;

	private static JPanel firstRow;

	private static Color[] presetColors = { new Color(255, 255, 255), new Color(192, 192, 192),
			new Color(213, 170, 213), new Color(170, 170, 255), new Color(170, 213, 255), new Color(170, 213, 170),
			new Color(255, 255, 170), new Color(250, 224, 175), new Color(255, 170, 170) };
	private static Color bgColor;
	private static int nbSlicesPreview;

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that allows to import the images stack with a 'maxFrames' frames
	 * claimed by the user.
	 * 
	 * @param imp       Stack of images opened in ImageJ
	 * @param maxFrames Number of frames wanted by the user (if 0 then all frames)
	 */
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
			} else if (imp.getType() == ImagePlus.GRAY16)
				dynamicRange = 16;
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

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that implements the user interface for the Background Plugin.
	 * 
	 * @return FIXME??
	 */
	protected static JFrame getGUIFrame() {

		processor = new ImageProcess();
		// Create a JFrame and add the JScrollPane to it
		frame = new JFrame("Background Removal"); // XXX ("TIFF Stack Preview");
		frame.setSize(1100, 750);
		frame.setLocationRelativeTo(null); // Center frame on screen
		// frame.setLayout(new FlowLayout());

		JPanel mainLayout = new JPanel();
		mainLayout.setPreferredSize(frame.getPreferredSize());

		JPanel loadFilePanel = createFileButton();
		mainLayout.add(loadFilePanel);

		previewPanel = getSecondRowPanel();
		processor.setPreviewPanel(previewPanel);
		processor.setBackgroundImagePanel(backgroundImagePanel); // FIXME
		processor.setSparseImagePanel(sparseImagePanel); // FIXME

		PreviewButtonActionListener previewButtonActionListener = new PreviewButtonActionListener(previewPanel,
				backgroundImagePanel);

		firstRow = getFirstRowPanel(previewButtonActionListener);

		// Load the spinner loading GIF
		ImageIcon loadingIcon = new ImageIcon(
				Objects.requireNonNull(BackgroundRemoval.class.getResource("/icons/loading.gif")));

		// Create the label with the icon
		loadingBackgroundLabel = new JLabel(loadingIcon);
		loadingSparseLabel = new JLabel(loadingIcon);
		loadingNoiseLabel = new JLabel(loadingIcon);

//		JButton previewButton = new JButton("Preview"); // TODO
//		previewButton.setPreferredSize(new Dimension(200, 30));

		JButton finalizeButton = new JButton("Finalize");
		finalizeButton.setPreferredSize(new Dimension(200, 30));

		// processor = new ImageProcess(previewPanel, backgroundImagePanel,
		// sparseImagePanel);
		chooseFileButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// Handle choose file button click
				FileDialog fd = new FileDialog(frame, "Choose Image", FileDialog.LOAD);
				fd.setVisible(true);
				String path = fd.getDirectory() + fd.getFile();
				// pathLabel.setMaximumSize(new Dimension(200,
				// pathLabel.getPreferredSize().height));
				pathLabel.setText(fd.getFile()); // Update path label text

				// TODO:check extension
				imp = IJ.openImage(path);

				importImage(imp, 0);

				// int index = previewPanel.getComponentZOrder(originalImagePanel);
				originalImagePanel.removeAll();
				originalImagePanel.add(MyGUI.getPreviewWindow(imp));

				originalImagePanel.revalidate();
				originalImagePanel.repaint();

				previewButtonActionListener.setImp(imp);
				previewButtonActionListener.setImageProcessor(processor);
			}
		});

		finalizeButton.addActionListener(new ActionListener() {
			@SuppressWarnings("unchecked")
			public void actionPerformed(ActionEvent e) {
				int stackSize = processor.getOriginalStackSize();
				processor.setStackSize(stackSize);

				/* Setting parameters */
				Component[] compo = firstRow.getComponents();
				System.err.println("Rank in finalize: " + ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
				processor.setRank((int) ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
				System.err.println("Power in finalize: " + ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
				processor.setPower((int) ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
				System.err.println("Err. tol. in finalize: " + ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
				processor.setTol((double) ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
				System.err.println("k in finalize: " + ((JSpinner) compo[2].getComponentAt(250, 47)).getValue());
				processor.setK((int) ((JSpinner) compo[2].getComponentAt(250, 47)).getValue());
				System.err.println("Tau in finalize: " + ((JSpinner) compo[2].getComponentAt(250, 7)).getValue());
				processor.setTau((double) ((JSpinner) compo[2].getComponentAt(250, 7)).getValue());
				System.err.println("Mode in finalize: "
						+ ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());
				processor.setMode((String) ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());

				SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
					@Override
					protected Void doInBackground() throws Exception {
						processor.finalize(imp);
						return null;
					}
				};
				worker.execute();
			}
		});
		mainLayout.add(firstRow);
		mainLayout.add(previewPanel);
		mainLayout.add(finalizeButton);

		frame.setLayout(new BoxLayout(frame.getContentPane(), BoxLayout.Y_AXIS));
		frame.add(mainLayout, BorderLayout.CENTER);
		// frame.add(previewPanel, BorderLayout.SOUTH);

		bgColor = presetColors[0];
		frame.setBackground(bgColor);

		return frame;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that creates the first 3 JPanels (first row) for the interface, the
	 * select file JPanel, Original Stack of images JPanel and Parameters JPanel.
	 * 
	 * @return JPanel that contains those 3 JPanels for the interface.
	 */
	protected static JPanel getFirstRowPanel(PreviewButtonActionListener previewButtonActionListener) {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		// TODO: replace choose file here...
		JPanel chooseFilePanel = new JPanel();

		chooseFilePanel.setPreferredSize(new Dimension(300, 300));
		chooseFilePanel.setLayout(new BorderLayout());
		chooseFilePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

		JPanel originalImage = new JPanel();

		originalImagePanel = new JPanel(null);
//		originalImagePanel.setPreferredSize(new Dimension(300, 300));
		originalImagePanel.setPreferredSize(new Dimension(300, 275));
		originalImagePanel.setLayout(new BorderLayout());
//		originalImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));
		originalImagePanel.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED));

		originalImage.add(originalImagePanel);

		// TODO: add parameters here...
		JPanel parametersPanel = new JPanel(null);
		parametersPanel.setPreferredSize(new Dimension(300, 275));
//		parametersPanel.setBorder(BorderFactory.createLineBorder(Color.red, 1));
//		parametersPanel.setBorder(BorderFactory.createEtchedBorder(Color.GRAY, Color.DARK_GRAY)); // Créer une bordure en relief
		parametersPanel.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED));

		Font font = new Font(Font.MONOSPACED, Font.PLAIN, 14);
		// Rank
		JLabel labelRank = new JLabel("Rank: ");
		labelRank.setFont(font);
		labelRank.setBounds(5, 5, 50, 30);
		SpinnerModel modelRank = new SpinnerNumberModel(3, 0, 99, 1);
		JSpinner spinnerRank = new JSpinner(modelRank);
		spinnerRank.setBounds(60, 7, 35, 30);
		parametersPanel.add(spinnerRank);
		parametersPanel.add(labelRank);
		// Power
		JLabel labelPower = new JLabel("Power: ");
		labelPower.setFont(font);
		labelPower.setBounds(5, 45, 75, 30);
		SpinnerModel modelPower = new SpinnerNumberModel(5, 0, 99, 1);
		JSpinner spinnerPower = new JSpinner(modelPower);
		spinnerPower.setBounds(60, 47, 35, 30);
		parametersPanel.add(spinnerPower);
		parametersPanel.add(labelPower);
		// Err tol
		JLabel labelTol = new JLabel("Err. tol.: ");
		labelTol.setFont(font);
		labelTol.setBounds(5, 85, 90, 30);
		SpinnerModel modelTol = new SpinnerNumberModel(0.001, 0, 99, 0.001);
		JSpinner spinnerTol = new JSpinner(modelTol);
		spinnerTol.setBounds(90, 87, 60, 30);
		parametersPanel.add(spinnerTol);
		parametersPanel.add(labelTol);
		// k
		JLabel labelK = new JLabel("k: ");
		labelK.setFont(font);
		labelK.setBounds(200, 45, 90, 30);
		SpinnerModel modelK = new SpinnerNumberModel(2, 0, 99, 1);
		JSpinner spinnerK = new JSpinner(modelK);
		spinnerK.setBounds(250, 47, 35, 30);
		parametersPanel.add(spinnerK);
		parametersPanel.add(labelK);
		// Tau
		JLabel labelTau = new JLabel("Tau: ");
		labelTau.setFont(font);
		labelTau.setBounds(200, 5, 90, 30);
		SpinnerModel modelTau = new SpinnerNumberModel(7.0, 0, 99, 1);
		JSpinner spinnerTau = new JSpinner(modelTau);
		spinnerTau.setBounds(250, 7, 35, 30);
		parametersPanel.add(spinnerTau);
		parametersPanel.add(labelTau);
		// Thresholding mode
		JLabel labelMode1 = new JLabel("Thresholding mode:");
		labelMode1.setFont(font);
		labelMode1.setBounds(5, 125, 150, 30);
		String[] pays = { "Soft", "Hard" };
		JComboBox<String> comboBox = new JComboBox<>(pays);
		comboBox.setBounds(160, 127, 75, 25);
		parametersPanel.add(labelMode1);
		parametersPanel.add(comboBox);

		JButton previewButton = new JButton("Preview"); // TODO
		previewButton.setBounds(100, 200, 100, 50); // setPreferredSize(new Dimension(200, 30));
		previewButton.addActionListener(previewButtonActionListener);
		previewButton.setToolTipText("Preview for the first 100 frames");
		parametersPanel.add(previewButton);

		rowPanel.add(chooseFilePanel);
		rowPanel.add(originalImage);
		rowPanel.add(parametersPanel);
		rowPanel.add(Box.createHorizontalStrut(50));

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that creates the 3 others JPanels (second row) for the interface, the
	 * Background Stack of images JPanel, Sparse Stack of images JPanel and Noise
	 * Stack of images JPanel.
	 * 
	 * @return JPanel that contains those 3 JPanels for the interface.
	 */
	protected static JPanel getSecondRowPanel() {
		JPanel previewPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		backgroundImagePanel = new JPanel();
		backgroundImagePanel.setPreferredSize(new Dimension(300, 300));
		backgroundImagePanel.setLayout(new BorderLayout());
		backgroundImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

		sparseImagePanel = new JPanel();
		sparseImagePanel.setPreferredSize(new Dimension(300, 300));
		sparseImagePanel.setLayout(new BorderLayout());
		sparseImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

		noiseImagePanel = new JPanel();
		noiseImagePanel.setPreferredSize(new Dimension(300, 300));
		noiseImagePanel.setLayout(new BorderLayout());
		noiseImagePanel.setBorder(BorderFactory.createLineBorder(Color.black, 1));

		rowPanel.add(backgroundImagePanel);
		rowPanel.add(sparseImagePanel);
		rowPanel.add(noiseImagePanel);
		rowPanel.add(Box.createHorizontalStrut(50));

		previewPanel.add(rowPanel);

		return previewPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that creates the Choose File/Select file button.
	 * 
	 * @return JPanel with this button
	 */
	protected static JPanel createFileButton() {
		JPanel panel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));
		panel.setPreferredSize(new Dimension(frame.getWidth() - 200, 30));

		JPanel buttonPanel = new JPanel();
		buttonPanel.setPreferredSize(new Dimension((int) (0.2 * panel.getPreferredSize().getWidth()),
				(int) panel.getPreferredSize().getHeight()));
		buttonPanel.setLayout(new BorderLayout());

		chooseFileButton = new JButton("Choose File");
		chooseFileButton.setPreferredSize(new Dimension(400, 30));

		buttonPanel.add(chooseFileButton, BorderLayout.CENTER);

		pathLabel = new JLabel("No file selected...");
		pathLabel.setPreferredSize(
				new Dimension((int) (0.5 * panel.getPreferredSize().getWidth()), pathLabel.getPreferredSize().height));
		pathLabel.setHorizontalAlignment(JLabel.LEFT);

		JPanel labelPanel = new JPanel();
		labelPanel.setPreferredSize(new Dimension((int) (0.5 * panel.getPreferredSize().getWidth()),
				(int) panel.getPreferredSize().getHeight()));
		labelPanel.setLayout(new BorderLayout());
		labelPanel.add(pathLabel, BorderLayout.CENTER);

		panel.add(buttonPanel);
		panel.add(Box.createHorizontalStrut(50));
		panel.add(labelPanel);

		return panel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * FIXME a completer
	 * 
	 * @param images
	 * @return
	 */
	protected static JPanel createPreviewWindow(ImagePlus[] images) {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));
		for (ImagePlus imp : images) {

			JPanel placeholder = new JPanel();
			placeholder.setPreferredSize(new Dimension(300, 300));
			placeholder.setLayout(new BorderLayout());
			placeholder.setBorder(BorderFactory.createLineBorder(Color.black, 1));
			placeholder.add(getPreviewWindow(imp), BorderLayout.CENTER);

			rowPanel.add(placeholder);

			rowPanel.revalidate();
			rowPanel.repaint();
		}

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that displays the preview of the Original Stack FIXME??
	 * 
	 * @param imp Images Stack
	 * @return
	 */
	protected static JPanel getPreviewWindow(ImagePlus imp) {
		if (imp != null) {
			// Get the number of frames in the TIFF stack
			int numFrames = imp.getStackSize();

			// Create a custom Canvas to display the preview
			StackCanvas canvas = new StackCanvas(imp);

			// Create a panel to hold the canvas with a fixed size
			JPanel placeholder = new JPanel(null);
			placeholder.setPreferredSize(new Dimension(300, 275));
			placeholder.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED));

			// Create a JSlider to slide frames manually
			JSlider slider = new JSlider(1, numFrames, 1);
			slider.setPreferredSize(
					new Dimension((int) placeholder.getPreferredSize().getWidth(), slider.getPreferredSize().height));

			slider.addChangeListener(new ChangeListener() {
				@Override
				public void stateChanged(ChangeEvent e) {
					int slice = slider.getValue();
					imp.setSlice(slice);
					canvas.repaint();
				}
			});

			// Create the first row JPanel and set its preferred size to take up 90% of the
			// space
			JPanel row1 = new JPanel(null);
			row1.setPreferredSize(new Dimension((int) placeholder.getPreferredSize().getWidth(),
					(int) (0.9 * placeholder.getPreferredSize().getHeight())));
			row1.setLayout(new BorderLayout());
			// Add the canvas to the center of the first row panel
			row1.add(canvas, BorderLayout.CENTER);

			// Create the second row JPanel and set its preferred size to take up 10% of the
			// space
			JPanel row2 = new JPanel();
			// row2.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
			row2.setPreferredSize(new Dimension((int) placeholder.getPreferredSize().getWidth(),
					(int) (0.1 * placeholder.getPreferredSize().getHeight())));
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

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * FIXME a completer
	 * 
	 * @return
	 */
	protected static JPanel createLoading() {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		JPanel backgroundImage = new JPanel();
		backgroundImage.setPreferredSize(new Dimension(300, 300));
		backgroundImage.setLayout(new BorderLayout());
		backgroundImage.setBorder(BorderFactory.createLineBorder(Color.black, 1));
		backgroundImage.add(loadingBackgroundLabel, BorderLayout.CENTER);

		JPanel sparseImage = new JPanel();
		sparseImage.setPreferredSize(new Dimension(300, 300));
		sparseImage.setLayout(new BorderLayout());
		sparseImage.setBorder(BorderFactory.createLineBorder(Color.black, 1));
		sparseImage.add(loadingSparseLabel, BorderLayout.CENTER);

		JPanel noiseImage = new JPanel();
		noiseImage.setPreferredSize(new Dimension(300, 300));
		noiseImage.setLayout(new BorderLayout());
		noiseImage.setBorder(BorderFactory.createLineBorder(Color.black, 1));
		noiseImage.add(loadingNoiseLabel, BorderLayout.CENTER);

		rowPanel.add(backgroundImage);
		rowPanel.add(sparseImage);
		rowPanel.add(noiseImage);
		rowPanel.add(Box.createHorizontalStrut(50));

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Class for the preview display.
	 */
	private static class StackCanvas extends Canvas {
		private static final long serialVersionUID = 1L;

		private final ImagePlus imp;
		private final int imageWidth;
		private final int imageHeight;

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Constructor of this class.
		 * 
		 * @param imp Stack of Images opened by ImageJ
		 */
		public StackCanvas(ImagePlus imp) {
			this.imp = imp;
			this.imageWidth = imp.getWidth();
			this.imageHeight = imp.getHeight();
//            setPreferredSize(new Dimension(imageWidth, imageHeight));
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Method that display the preview.
		 */
		@Override
		public void paint(Graphics g) {
			// Create off-screen buffer
			Image offscreen = createImage(getWidth(), getHeight() + 30);
			Graphics buffer = offscreen.getGraphics();

			// Calculate the scale factor for the image
			double scale = Math.min((double) getWidth() / imp.getWidth(), (double) getHeight() / imp.getHeight());

//			if (imp.getWidth() < getWidth() && imp.getHeight() < getHeight()) {
//				System.err.println("CE FA");
//				scale = 1.0;
//			}

			// Get the current slice and the corresponding image processor
			int slice = imp.getCurrentSlice();
			FileInfo fi = imp.getFileInfo();
			BufferedImage img = imp.getStack().getProcessor(slice).getBufferedImage();

			int scaledWidth = 300;// (int) (imp.getWidth() * scale);
			int scaledHeight = 275; // (int) (imp.getHeight() * scale);

			// Calculate the offset needed to center the image on the canvas
			int offsetX = (getWidth() - scaledWidth) / 2;
			int offsetY = (getHeight() - scaledHeight) / 2;

			// Draw the image on the off-screen buffer
			buffer.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight, null);

			// TODO:replace this sliceText

			// Draw the slice number and file name in the bottom-left corner
			String sliceText = "Slice: " + slice + "/" + imp.getStackSize();
			buffer.drawString(sliceText, 10, getHeight() + 30);

			// Swap the buffers
			g.drawImage(offscreen, 0, 0, getWidth(), getHeight(), null);
		}
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Class for the preview action.
	 */
	public static class PreviewButtonActionListener implements ActionListener {
		private JPanel backgroundImagePanel;
		private final JPanel previewPanel;
		private ImagePlus imp;
		private ImageProcess processor;

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Constructor of this class.
		 * 
		 * @param previewPanel
		 * @param backgroundImagePanel
		 */
		public PreviewButtonActionListener(JPanel previewPanel, JPanel backgroundImagePanel) {
			this.backgroundImagePanel = backgroundImagePanel;
			this.previewPanel = previewPanel;
//            this.processor = new ImageProcess();
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Method that sets the stack of images.
		 * 
		 * @param imp
		 */
		public void setImp(ImagePlus imp) {
			this.imp = imp;
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Method that sets the processor that runs the plugin.
		 * 
		 * @param processor
		 */
		public void setImageProcessor(ImageProcess processor) {
			this.processor = processor;
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Method that does the preview of the Background Removal plugin for the first
		 * 100 frames or the size of the stack if the total number of frames is less
		 * than 100.
		 */
		@SuppressWarnings("unchecked")
		public void actionPerformed(ActionEvent e) {

			processor.setStackSize(nbSlicesPreview);

			/* Setting parameters */
			Component[] compo = firstRow.getComponents();
			System.err.println("Rank in preview: " + ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
			processor.setRank((int) ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
			System.err.println("Power in preview: " + ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
			processor.setPower((int) ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
			System.err.println("Err. tol. in preview: " + ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
			processor.setTol((double) ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
			System.err.println("k in preview: " + ((JSpinner) compo[2].getComponentAt(250, 47)).getValue());
			processor.setK((int) ((JSpinner) compo[2].getComponentAt(250, 47)).getValue());
			System.err.println("Tau in preview: " + ((JSpinner) compo[2].getComponentAt(250, 7)).getValue());
			processor.setTau((double) ((JSpinner) compo[2].getComponentAt(250, 7)).getValue());
			System.err.println(
					"Mode in preview: " + ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());
			processor.setMode((String) ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());

			SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
				@Override
				protected Void doInBackground() throws Exception {
					processor.preview(imp);
					return null;
				}
			};
			worker.execute();
		}
	}
}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
