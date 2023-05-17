package com.mycompany.imagej;

import ij.IJ;
import ij.ImagePlus;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
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

	/** Informations about the Images Stack */
	private static int IMGWidth, IMGHeight, IMGnbSlices;

	/** True if already a file in GUI */
	private static boolean alreadySelected;

	/** Progress label in GUI */
	private static JProgressBar progress;

	/** List of all sliders */
	private static ArrayList<JSlider> sliders = new ArrayList<>();

	/** Level of estimated time */
	private static JLabel labelEstimedTime;

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
		IMGWidth = ImgWidth;
		processor.setHeight(ImgHeight);
		IMGHeight = ImgHeight;
		processor.setStackSize(nbSlices);
		IMGnbSlices = nbSlices;
		processor.setOriginalStackSize(stackSize);
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that implements the user interface for the Background Plugin.
	 * 
	 * @return the GUI frame
	 */
	protected static JFrame getGUIFrame() {
		alreadySelected = false;

		processor = new ImageProcess();
		// Create a JFrame and add the JScrollPane to it
		frame = new JFrame("Background Removal"); 
		frame.setSize(1100, 750);
		frame.setLocationRelativeTo(null); // Center frame on screen
		// frame.setLayout(new FlowLayout());

		JPanel mainLayout = new JPanel();
		mainLayout.setPreferredSize(frame.getPreferredSize());

		previewPanel = getSecondRowPanel();
		processor.setPreviewPanel(previewPanel);
		processor.setBackgroundImagePanel(backgroundImagePanel); 
		processor.setSparseImagePanel(sparseImagePanel); 

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

		labelEstimedTime = new JLabel(new ImageIcon(
				(new ImageIcon(Objects.requireNonNull(BackgroundRemoval.class.getResource("/icons/gray_clock.png")))
						.getImage()).getScaledInstance(40, 40, Image.SCALE_SMOOTH)));

		progress = new JProgressBar(0, 100);
		progress.setValue(0);
		progress.setStringPainted(true);

		Font font = new Font(Font.MONOSPACED, Font.PLAIN, 14);
		JButton finalizeButton = new JButton("Finalize");
		finalizeButton.setFont(font);
		finalizeButton.setPreferredSize(new Dimension(200, 30));

		// Rank
		JLabel labelBeforeFinalize = new JLabel("Click on Finalize to complete the processing -> ");
		labelBeforeFinalize.setFont(font);

		Component[] compo = firstRow.getComponents();
		JPanel lastPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));

		chooseFileButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// Handle choose file button click
				FileDialog fd = new FileDialog(frame, "Choose Images Stack", FileDialog.LOAD);
				fd.setVisible(true);
				String path = fd.getDirectory() + fd.getFile();
				pathLabel.setText(fd.getFile()); // Update path label text

				// TODO:check extension
				imp = IJ.openImage(path);

				importImage(imp, 0);

				originalImagePanel.removeAll();

				originalImagePanel.add(MyGUI.getPreviewWindow(imp));

				originalImagePanel.revalidate();
				originalImagePanel.repaint();

				if (alreadySelected) {
					previewPanel.removeAll();
					previewPanel.add(MyGUI.createWaitingChooseFile());

					previewPanel.revalidate();
					previewPanel.repaint();

					progress.setValue(0);
				}

				previewButtonActionListener.setImp(imp);
				previewButtonActionListener.setImageProcessor(processor);

				alreadySelected = true;

				lastPanel.removeAll();

				// TODO a ameliorer
				if (IMGWidth * IMGHeight * IMGnbSlices <= 5_000_000) {
					labelEstimedTime = new JLabel(new ImageIcon((new ImageIcon(
							Objects.requireNonNull(BackgroundRemoval.class.getResource("/icons/green_clock.png")))
							.getImage()).getScaledInstance(40, 40, Image.SCALE_SMOOTH)));
				} else if (IMGWidth * IMGHeight * IMGnbSlices <= 50_000_000) {
					labelEstimedTime = new JLabel(new ImageIcon((new ImageIcon(
							Objects.requireNonNull(BackgroundRemoval.class.getResource("/icons/orange_clock.png")))
							.getImage()).getScaledInstance(40, 40, Image.SCALE_SMOOTH)));
				} else {
					labelEstimedTime = new JLabel(new ImageIcon((new ImageIcon(
							Objects.requireNonNull(BackgroundRemoval.class.getResource("/icons/red_clock.png")))
							.getImage()).getScaledInstance(40, 40, Image.SCALE_SMOOTH)));
				}

				lastPanel.add(progress);
				lastPanel.add(Box.createHorizontalStrut(50));
				lastPanel.add(labelBeforeFinalize);
				lastPanel.add(finalizeButton);
				lastPanel.add(Box.createHorizontalStrut(40));
				lastPanel.add(labelEstimedTime);
			}
		});

		finalizeButton.addActionListener(new ActionListener() {
			@SuppressWarnings("unchecked")
			public void actionPerformed(ActionEvent e) {
				int stackSize = processor.getOriginalStackSize();
				processor.setStackSize(stackSize);

				/* Setting parameters */
				processor.setRank((int) ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
				processor.setPower((int) ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
				processor.setTol((double) ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
				processor.setK((int) ((JSpinner) compo[2].getComponentAt(200, 47)).getValue());
				processor.setTau((double) ((JSpinner) compo[2].getComponentAt(200, 7)).getValue());
				processor.setMode((String) ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());

				progress.setValue(0);

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

		lastPanel.add(progress);
		lastPanel.add(Box.createHorizontalStrut(50));
		lastPanel.add(labelBeforeFinalize);
		lastPanel.add(finalizeButton);
		lastPanel.add(Box.createHorizontalStrut(40));
		lastPanel.add(labelEstimedTime);

		mainLayout.add(firstRow);
		mainLayout.add(previewPanel);
		mainLayout.add(lastPanel);

		frame.setLayout(new BoxLayout(frame.getContentPane(), BoxLayout.Y_AXIS));
		frame.add(mainLayout, BorderLayout.CENTER);

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

		JPanel loadFilePanel = createFileButton();

		JPanel originalImage = new JPanel();

		originalImagePanel = new JPanel(new BorderLayout());
		originalImagePanel.setPreferredSize(new Dimension(300, 300));
		originalImagePanel
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Original", TitledBorder.CENTER, TitledBorder.TOP));
		originalImage.add(originalImagePanel);

		/** Parametres frame */
		JPanel parametersPanel = new JPanel(null);
		parametersPanel.setPreferredSize(new Dimension(290, 275));
		parametersPanel.setBorder(BorderFactory.createLoweredBevelBorder());

		Font font = new Font(Font.MONOSPACED, Font.PLAIN, 14);
		// Rank
		JLabel labelRank = new JLabel("Rank: ");
		labelRank.setFont(font);
		labelRank.setBounds(5, 5, 50, 30);
		SpinnerModel modelRank = new SpinnerNumberModel(3, 0, 99, 1);
		JSpinner spinnerRank = new JSpinner(modelRank);
		spinnerRank.setBounds(60, 7, 50, 30);
		parametersPanel.add(spinnerRank);
		parametersPanel.add(labelRank);
		// Power
		JLabel labelPower = new JLabel("Power: ");
		labelPower.setFont(font);
		labelPower.setBounds(5, 45, 75, 30);
		SpinnerModel modelPower = new SpinnerNumberModel(5, 0, 99, 1);
		JSpinner spinnerPower = new JSpinner(modelPower);
		spinnerPower.setBounds(60, 47, 50, 30);
		parametersPanel.add(spinnerPower);
		parametersPanel.add(labelPower);
		// Err tol
		JLabel labelTol = new JLabel("Err. tol.: ");
		labelTol.setFont(font);
		labelTol.setBounds(5, 85, 90, 30);
		SpinnerModel modelTol = new SpinnerNumberModel(0.001, 0, 99, 0.001);
		JSpinner spinnerTol = new JSpinner(modelTol);
		spinnerTol.setBounds(90, 87, 90, 30);
		parametersPanel.add(spinnerTol);
		parametersPanel.add(labelTol);
		// k
		JLabel labelK = new JLabel("k: ");
		labelK.setFont(font);
		labelK.setBounds(150, 45, 90, 30);
		SpinnerModel modelK = new SpinnerNumberModel(2, 0, 99, 1);
		JSpinner spinnerK = new JSpinner(modelK);
		spinnerK.setBounds(200, 47, 60, 30);
		parametersPanel.add(spinnerK);
		parametersPanel.add(labelK);
		// Tau
		JLabel labelTau = new JLabel("Tau: ");
		labelTau.setFont(font);
		labelTau.setBounds(150, 5, 90, 30);
		SpinnerModel modelTau = new SpinnerNumberModel(7.0, 0, 99, 1);
		JSpinner spinnerTau = new JSpinner(modelTau);
		spinnerTau.setBounds(200, 7, 60, 30);
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

		JButton previewButton = new JButton("Preview"); 
		previewButton.setFont(font);
		previewButton.setBounds(85, 200, 125, 30); // setPreferredSize(new Dimension(200, 30));
		previewButton.addActionListener(previewButtonActionListener);
		previewButton.setToolTipText("Preview for the first 100 frames");
		parametersPanel.add(previewButton);

		rowPanel.add(loadFilePanel);
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
		JPanel previewPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 25));

		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		backgroundImagePanel = new JPanel();
		backgroundImagePanel.setPreferredSize(new Dimension(300, 300));
		backgroundImagePanel.setLayout(new BorderLayout());
		backgroundImagePanel
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Background", TitledBorder.CENTER, TitledBorder.TOP));

		sparseImagePanel = new JPanel();
		sparseImagePanel.setPreferredSize(new Dimension(300, 300));
		sparseImagePanel.setLayout(new BorderLayout());
		sparseImagePanel
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Sparse", TitledBorder.CENTER, TitledBorder.TOP));

		noiseImagePanel = new JPanel();
		noiseImagePanel.setPreferredSize(new Dimension(300, 300));
		noiseImagePanel.setLayout(new BorderLayout());
		noiseImagePanel
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Noise", TitledBorder.CENTER, TitledBorder.TOP));

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
		JPanel panel = new JPanel(null);
		panel.setPreferredSize(new Dimension(300, 275));

		JPanel buttonPanel = new JPanel(null);
		buttonPanel.setPreferredSize(new Dimension(300, 275));

		Font font = new Font(Font.MONOSPACED, Font.PLAIN, 14);

		chooseFileButton = new JButton("Select file...");
		chooseFileButton.setFont(font);
		chooseFileButton.setBounds(50, 100, 200, 30);

		panel.add(chooseFileButton);

		pathLabel = new JLabel("No file selected...");
		pathLabel.setFont(font);
		pathLabel.setBounds(50, 150, 175, 30);
		panel.add(pathLabel);

		JPanel labelPanel = new JPanel();
		labelPanel.setPreferredSize(new Dimension((int) (0.5 * panel.getPreferredSize().getWidth()),
				(int) panel.getPreferredSize().getHeight()));
		labelPanel.setLayout(new BorderLayout());

		panel.add(buttonPanel);
		panel.add(Box.createHorizontalStrut(50));
		panel.add(labelPanel);

		return panel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Function
	 * 
	 * @param images
	 * @return
	 */
	protected static JPanel createPreviewWindow(ImagePlus[] images) {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		int i = 0;
		for (ImagePlus imp : images) {

			JPanel placeholder = new JPanel();
			placeholder.setPreferredSize(new Dimension(300, 300));
			placeholder.setLayout(new BorderLayout());
			if (i == 0) {
				placeholder.setBorder(
						BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
								"Background", TitledBorder.CENTER, TitledBorder.TOP));
			} else if (i == 1) {
				placeholder.setBorder(
						BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
								"Sparse", TitledBorder.CENTER, TitledBorder.TOP));
			} else {
				placeholder.setBorder(
						BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
								"Noise", TitledBorder.CENTER, TitledBorder.TOP));
			}

			placeholder.add(getPreviewWindow(imp), BorderLayout.CENTER);

			rowPanel.add(placeholder);

			rowPanel.revalidate();
			rowPanel.repaint();
			i++;
		}
		rowPanel.add(Box.createHorizontalStrut(50));

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that displays the preview screens.
	 * 
	 * @param imp Images Stack
	 * @return
	 */
	protected static JPanel getPreviewWindow(ImagePlus imp) {
		if (imp != null) {

			ImagePlusPanel panel = new ImagePlusPanel(imp);
			panel.setPreferredSize(new Dimension(400, 400));

			JPanel placeholder = new JPanel(null);
			placeholder.setPreferredSize(new Dimension(300, 275));

			JSlider slider = new JSlider(1, imp.getStackSize(), 1);
			sliders.add(slider);
			slider.setPreferredSize(
					new Dimension((int) placeholder.getPreferredSize().getWidth(), slider.getPreferredSize().height));

			JLabel sliceInfoLabel = new JLabel();

			String sliceText = "Slice: " + slider.getValue() + "/" + imp.getStackSize();
			sliceInfoLabel.setText(sliceText);

			slider.addChangeListener(new ChangeListener() {
				@Override
				public void stateChanged(ChangeEvent e) {
					synchronizeSliders(slider, sliders);
					int slice = slider.getValue();
					imp.setSlice(slice);
					Image img = imp.getProcessor().getBufferedImage();
					panel.setImage(img);

					String sliceText = "Slice: " + slice + "/" + imp.getStackSize();
					sliceInfoLabel.setText(sliceText);
				}
			});

			JPanel row1 = new JPanel(null);
			row1.setPreferredSize(new Dimension((int) placeholder.getPreferredSize().getWidth(),
					(int) (0.85 * placeholder.getPreferredSize().getHeight())));
			row1.setLayout(new BorderLayout());
			// Add the canvas to the center of the first row panel
			row1.add(panel, BorderLayout.CENTER);

			JPanel row2 = new JPanel();
			row2.setPreferredSize(new Dimension((int) placeholder.getPreferredSize().getWidth(),
					(int) (0.15 * placeholder.getPreferredSize().getHeight())));
			row2.setLayout(new BorderLayout());
			// Add the slider to the center of the second row panel
			row2.add(sliceInfoLabel, BorderLayout.WEST);
			row2.add(slider, BorderLayout.SOUTH);

			placeholder.setLayout(new BoxLayout(placeholder, BoxLayout.PAGE_AXIS));
			placeholder.add(row1);
			placeholder.add(row2);

			return placeholder;
		}
		return null;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that show the loading process.
	 * 
	 * @return JPanel with the loading .gif
	 */
	protected static JPanel createLoading() {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		JPanel backgroundImage = new JPanel();
		backgroundImage.setPreferredSize(new Dimension(300, 300));
		backgroundImage.setLayout(new BorderLayout());
		backgroundImage
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Background", TitledBorder.CENTER, TitledBorder.TOP));
		backgroundImage.add(loadingBackgroundLabel, BorderLayout.CENTER);

		JPanel sparseImage = new JPanel();
		sparseImage.setPreferredSize(new Dimension(300, 300));
		sparseImage.setLayout(new BorderLayout());
		sparseImage.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
				"Sparse", TitledBorder.CENTER, TitledBorder.TOP));
		sparseImage.add(loadingSparseLabel, BorderLayout.CENTER);

		JPanel noiseImage = new JPanel();
		noiseImage.setPreferredSize(new Dimension(300, 300));
		noiseImage.setLayout(new BorderLayout());
		noiseImage.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
				"Noise", TitledBorder.CENTER, TitledBorder.TOP));
		noiseImage.add(loadingNoiseLabel, BorderLayout.CENTER);

		rowPanel.add(backgroundImage);
		rowPanel.add(sparseImage);
		rowPanel.add(noiseImage);
		rowPanel.add(Box.createHorizontalStrut(50));

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that show the JPanels while the user choose another file.
	 * 
	 * @return JPanels empty.
	 */
	protected static JPanel createWaitingChooseFile() {
		JPanel rowPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));

		JPanel backgroundImage = new JPanel();
		backgroundImage.setPreferredSize(new Dimension(300, 300));
		backgroundImage.setLayout(new BorderLayout());
		backgroundImage
				.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
						"Background", TitledBorder.CENTER, TitledBorder.TOP));

		JPanel sparseImage = new JPanel();
		sparseImage.setPreferredSize(new Dimension(300, 300));
		sparseImage.setLayout(new BorderLayout());
		sparseImage.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
				"Sparse", TitledBorder.CENTER, TitledBorder.TOP));

		JPanel noiseImage = new JPanel();
		noiseImage.setPreferredSize(new Dimension(300, 300));
		noiseImage.setLayout(new BorderLayout());
		noiseImage.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED),
				"Noise", TitledBorder.CENTER, TitledBorder.TOP));

		rowPanel.add(backgroundImage);
		rowPanel.add(sparseImage);
		rowPanel.add(noiseImage);
		rowPanel.add(Box.createHorizontalStrut(50));

		return rowPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Methods that updates the progress bar of the GreGoDec algorithm.
	 * 
	 * @param value
	 */
	public static void setProgressBar(int value) {
		progress.setValue(value);
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
			processor.setRank((int) ((JSpinner) compo[2].getComponentAt(60, 7)).getValue());
			processor.setPower((int) ((JSpinner) compo[2].getComponentAt(60, 47)).getValue());
			processor.setTol((double) ((JSpinner) compo[2].getComponentAt(90, 87)).getValue());
			processor.setK((int) ((JSpinner) compo[2].getComponentAt(200, 47)).getValue());
			processor.setTau((double) ((JSpinner) compo[2].getComponentAt(200, 7)).getValue());
			processor.setMode((String) ((JComboBox<String>) compo[2].getComponentAt(160, 127)).getSelectedItem());

			progress.setValue(0);

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

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that synchronize all the sliders in the interface.
	 * 
	 * @param sourceSlider slider that changed
	 * @param sliders      all sliders in the interface
	 */
	private static void synchronizeSliders(JSlider sourceSlider, ArrayList<JSlider> sliders) {
		int value = sourceSlider.getValue();
		for (JSlider slider : sliders) {
			if (slider != sourceSlider) {
				slider.setValue(value);
			}
		}
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/*-----------------------------------------------------------------------------------------------------------------------*/
	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Class for the slices display.
	 */
	public static class ImagePlusPanel extends JPanel {
		private static final long serialVersionUID = 1L;
		private ImagePlus imp;
		private Image img;

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * ImagePlusPanel unique constructor.
		 * 
		 * @param imp images stack
		 */
		public ImagePlusPanel(ImagePlus imp) {
			this.imp = imp;
			this.img = imp.getImage();
			setPreferredSize(new Dimension(imp.getWidth(), imp.getHeight()));
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Method that sets the images stack.
		 * 
		 * @param img images stack
		 */
		public void setImage(Image img) {
			this.img = img;
			repaint();
		}

		/*-----------------------------------------------------------------------------------------------------------------------*/
		/**
		 * Methods used to repaint slice.
		 */
		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			double scale = Math.min((double) getWidth() / imp.getWidth(), (double) getHeight() / imp.getHeight());

			int scaledWidth = (int) (imp.getWidth() * scale);
			int scaledHeight = (int) (imp.getHeight() * scale);

			int offsetX = (getWidth() - scaledWidth) / 2;
			int offsetY = (getHeight() - scaledHeight) / 2;

			g.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight, null);
		}
	}
}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
