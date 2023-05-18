package com.mycompany.imagej;

import ij.ImagePlus;
import net.imglib2.type.numeric.RealType;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.io.File;

/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/**
 * Class with the main method used by the ImageJ plugin.
 *
 * @param <T>
 */
@Plugin(type = Command.class, menuPath = "Plugins>Background Removal")
public class BackgroundRemoval<T extends RealType<T>> implements Command {

	static Color[] presetColors = { new Color(255, 255, 255), new Color(192, 192, 192), new Color(213, 170, 213),
			new Color(170, 170, 255), new Color(170, 213, 255), new Color(170, 213, 170), new Color(255, 255, 170),
			new Color(250, 224, 175), new Color(255, 170, 170) };
	static Color bgColor;

	private static MyGUI.PreviewButtonActionListener previewButtonActionListener;
	private static JButton chooseFileButton;
	private static JLabel pathLabel;
	private static JPanel originalImagePanel;
	private static JPanel backgroundImagePanel;
	private static JPanel sparseImagePanel;
	private static JPanel noiseImagePanel;
	private static JPanel previewPanel;
	private static JFrame frame;
	private static ImagePlus imp;
	JButton previewButton;

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * The main function that executes the plugin.
	 * 
	 * @param args none
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {
		execute();
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Function called in the main function to generate the interface and start the
	 * plugin.
	 */
	public static void execute() {
		generateInterface();
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Function used before to check the file extension in logs
	 * 
	 * @param file File used
	 */
	public static void checkFileExtension(File file) {
		String fileExtension = getFileExtension(file);

		if (fileExtension.equals("tif") || fileExtension.equals("tiff")) {
			System.out.println("TIF stack loading OK");
		} else {
			throw new RuntimeException("The file extension should be .tif, .tiff");
		}
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Function used before to get the extension in the file name.
	 * 
	 * @param file File used
	 * @return
	 */
	public static String getFileExtension(File file) {
		String fileName = file.getName();
		if (fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0)
			return fileName.substring(fileName.lastIndexOf(".") + 1);
		else
			return "";
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Generate the user interface and start the plugin.
	 */
	public static void generateInterface() {
		// Create a JFrame and add the JScrollPane to it
		frame = MyGUI.getGUIFrame();

		// frame.pack();
		frame.setVisible(true);

		// Pour empecher la modificaiton de la fenetre
		frame.setResizable(false);

	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	@Override
	public void run() {
		execute();
	}
}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
