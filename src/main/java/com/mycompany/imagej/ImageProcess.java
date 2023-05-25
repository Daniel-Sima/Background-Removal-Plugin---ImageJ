package com.mycompany.imagej;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.ejml.simple.SimpleMatrix;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Random;

/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/**
 * Class with the Background Removal computation methods.
 *
 */
public class ImageProcess {

	private static int dynamicRange = 8, k = 2, rank = 3, power = 5, height, width, stackSize, originalStackSize;
	private static double tau = 7;
	private static MODE mode = MODE.SOFT;
	private static double tol = 0.001;

	private static JPanel previewPanel;
	private static JPanel backgroundImagePanel;
	private static JPanel sparseImagePanel;

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that vectorize the Stack of Images in a DenseMatrix64F.
	 * 
	 * @param stack Stack of Images opened by ImageJ
	 * @param bit   8 for GRAY8 / 16 for GRAY16
	 * @return The vectorized matrix
	 */
	private static DenseMatrix64F constructMatrix(ImageStack stack, int bit) {

		DenseMatrix64F denseMatrix = new DenseMatrix64F(width * height, stackSize);

		if (bit == 8) {
			for (int z = 0; z < stackSize; z++) {
				ByteProcessor bp = (ByteProcessor) stack.getProcessor(z + 1);
				int index = 0;
				for (int i = 0; i < height; i++) {
					for (int j = 0; j < width; j++) {
						denseMatrix.set(index++, z, bp.getPixelValue(j, i));
					}
				}
			}

		}
		return denseMatrix;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that does the SVD computation randomly, to gain in performance.
	 * 
	 * @param A Matrix to apply the SVD
	 * @param k Number of biggest first singular values
	 * @return U, S and V matrix after the SVD computation.
	 */
	public static ArrayList<DenseMatrix64F> randomizedSVD(DenseMatrix64F A, int k) {
		int n = A.numCols;

		// Etape 1: Generer une matrice aleatoire R de taille n x k
		// imperativement aleatoire entre 0 et 1
		SimpleMatrix RR = SimpleMatrix.random(n, k, 0, 1, new java.util.Random());

		DenseMatrix64F R = new DenseMatrix64F(n, k);
		for (int x = 0; x < n; x++) {
			for (int z = 0; z < k; z++) {
				R.set(x, z, RR.get(x, z));
			}
		}
		
		// Etape 2: Calculer le produit matriciel Y = A * R
		DenseMatrix64F Y = new DenseMatrix64F(A.numRows, R.numCols);
		CommonOps.mult(A, R, Y);

		// Etape 3: Effectuer une decomposition QR sur la matrice Y
		QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(Y.numRows, Y.numCols);
		double[][] data = new double[Y.numRows][Y.numCols];
		for (int i = 0; i < Y.numRows; i++) {
			for (int j = 0; j < Y.numCols; j++) {
				data[i][j] = Y.get(i, j);
			}
		}
		qr.decompose(new DenseMatrix64F(data));
		DenseMatrix64F Qs = qr.getQ(null, true);

		// Etape 4: Calculer la matrice B = Q^T * A
		DenseMatrix64F Q = new DenseMatrix64F(Qs.numCols, Qs.numRows);
		CommonOps.transpose(Qs, Q);
		DenseMatrix64F B = new DenseMatrix64F(Q.numRows, A.numCols);
		CommonOps.mult(Q, A, B);

		// Etape 5: Appliquer la SVD sur la matrice B
		SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(B.numRows, B.numCols, true, true,
				true);
		DenseMatrix64F BB = new DenseMatrix64F(B.numRows, B.numCols);
		for (int i = 0; i < B.numRows; i++) {
			for (int j = 0; j < B.numCols; j++) {
				BB.set(i, j, B.get(i, j));
			}
		}
		svd.decompose(BB);

		DenseMatrix64F U_ss = svd.getU(null, false);
		DenseMatrix64F S_ss = svd.getW(null);
		DenseMatrix64F V_ss = svd.getV(null, false);

		// Etape 6: Calculer la matrice U de la decomposition en valeurs singulieres
		// tronquees de la matrice d'entree A
		DenseMatrix64F Us = new DenseMatrix64F(Qs.numRows, U_ss.numCols);
		CommonOps.mult(Qs, U_ss, Us);

		DenseMatrix64F V_s = new DenseMatrix64F(V_ss.numCols, V_ss.numRows);
		CommonOps.transpose(V_ss, V_s);

		ArrayList<DenseMatrix64F> result = new ArrayList<>();
		result.add(Us);
		result.add(V_s);
		result.add(S_ss);

		// Renvoyer les matrices U, S et V de la Truncated SVD
		return result;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that does the threshold, similar to the threshold method in PyWavelets
	 * from Python.
	 * 
	 * @param data Matrix to apply the threshold
	 * @param tau
	 * @param mode
	 * @return Matrix with the applied threshold
	 */
	public static DenseMatrix64F threshold(DenseMatrix64F data, double tau, String mode) {
		int rows = data.numRows;
		int cols = data.numCols;
		DenseMatrix64F result = new DenseMatrix64F(rows, cols);
		if (mode.equals("SOFT")) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double val = data.get(i, j);
					if (Math.abs(val) < tau) {
						result.set(i, j, 0);
					} else {
						result.set(i, j, (val / Math.abs(val) * Math.max(Math.abs(val) - tau, 0)));
					}
				}
			}
		} else if (mode.equals("HARD")) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double val = data.get(i, j);
					if (Math.abs(val) < tau) {
						result.set(i, j, 0);
					} else {
						result.set(i, j, val);
					}
				}
			}
		} else {
			System.out.println("mode not supported");
			throw new RuntimeException("mode not supported");
		}
		return result;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that does the mean of the Matrix, i.e adds all the elements and divide
	 * by the number of elements of the Matrix betwen 'start' and 'end' lines.
	 * 
	 * @param matrix
	 * @param start
	 * @param end
	 * @return
	 */
	private static double mean(DenseMatrix64F matrix, int start, int end) {
		double sum = 0;
		int count = 0;
		for (int i = start; i < end; i++) {
			for (int j = 0; j < matrix.numCols; j++) {
				sum += matrix.get(i, j);
				count++;
			}
		}
		return sum / count;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that reconstructs the Stack of Images from a Matrix and does the
	 * normalization.
	 * 
	 * @param matrix
	 * @param bit
	 * @return
	 */
	private static ImageStack constructImageStack(DenseMatrix64F matrix, int bit) {

		ImageStack newStack = new ImageStack(width, height);

		double[][] array = new double[matrix.numCols][matrix.numRows];

		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;

		for (int r = 0; r < matrix.numRows; r++) {
			for (int c = 0; c < matrix.numCols; c++) {
				array[c][r] = matrix.get(r, c);
				// trouver la valeur minimale et la valeur maximale pour normalization
				min = Math.min(min, matrix.get(r, c));
				max = Math.max(max, matrix.get(r, c));
			}
		}

		// normaliser les données
		for (int i = 0; i < array.length; i++) {
			byte[] pixels = new byte[width * height];
			for (int j = 0; j < array[i].length; j++) {
				array[i][j] = ((array[i][j] - min) / (max - min)) * (Math.pow(2, dynamicRange) - 1);
				pixels[j] = (byte) Math.round(array[i][j]);
			}
			ByteProcessor bp = new ByteProcessor(width, height, pixels);

			newStack.addSlice(bp);
		}

		return newStack;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the Dynamic Range parameter
	 * 
	 * @param type (8 for GRAY8; 16 for GRAY16)
	 */
	public void setDynamicRange(int type) {
		dynamicRange = type;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'k' parameter, i.e the number of firsts bigger singular
	 * values.
	 * 
	 * @param k
	 */
	public void setK(int k) {
		ImageProcess.k = k;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'tau' value for the thresholding.
	 * 
	 * @param tau
	 */
	public void setTau(double tau) {
		ImageProcess.tau = tau;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'mode' value for the thresholding.
	 * 
	 * @param mode
	 */
	public void setMode(String mode) {
		if (mode == "Hard") {
			ImageProcess.mode = MODE.HARD;
		}

		if (mode == "Soft") {
			ImageProcess.mode = MODE.SOFT;
		}
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'rank' parameter for the GreGoDec algorithm.
	 * 
	 * @param rank
	 */
	public void setRank(int rank) {
		ImageProcess.rank = rank;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'power' parameter for the GreGoDec algorithm.
	 * 
	 * @param power
	 */
	public void setPower(int power) {
		ImageProcess.power = power;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the 'tol' parameter for the error margin in the GreGoDec
	 * algorithm.
	 * 
	 * @param tol
	 */
	public void setTol(double tol) {
		ImageProcess.tol = tol;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the width of the images in stack.
	 * 
	 * @param width Images width
	 */
	public void setWidth(int width) {
		ImageProcess.width = width;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the height of the images in stack.
	 * 
	 * @param height Images height
	 */
	public void setHeight(int height) {
		ImageProcess.height = height;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the size of the stack wanted by the user.
	 * 
	 * @param stackSize Number of frames wanted in stack
	 */
	public void setStackSize(int stackSize) {
		ImageProcess.stackSize = stackSize;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that gets the original stack size.
	 * 
	 * @return originalStackSize
	 */
	public int getOriginalStackSize() {
		return ImageProcess.originalStackSize;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the **full** size of the stack.
	 * 
	 * @param stackSize Number of frames in stack
	 */
	public void setOriginalStackSize(int stackSize) {
		ImageProcess.originalStackSize = stackSize;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method thats sets the interface JPanel.
	 * 
	 * @param previewPanel
	 */
	public void setPreviewPanel(JPanel previewPanel) {
		ImageProcess.previewPanel = previewPanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the Background JPanel
	 * 
	 * @param backgroundImagePanel
	 */
	public void setBackgroundImagePanel(JPanel backgroundImagePanel) {
		ImageProcess.backgroundImagePanel = backgroundImagePanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that sets the Sparse JPanel
	 * 
	 * @param sparseImagePanel
	 */
	public void setSparseImagePanel(JPanel sparseImagePanel) {
		ImageProcess.sparseImagePanel = sparseImagePanel;
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that shows the result of the Background Removal on the JPanels.
	 * 
	 * @param imp Images Stack opened by ImageJ
	 */
	public void finalize(ImagePlus imp) {
		previewPanel.removeAll();
		previewPanel.add(MyGUI.createLoading());

		previewPanel.revalidate();
		previewPanel.repaint();

		ImagePlus[] images = process(imp);

		previewPanel.removeAll();
		previewPanel.add(MyGUI.createPreviewWindow(images));

		previewPanel.revalidate();
		previewPanel.repaint();

		images[0].show();
		images[1].show();
		images[2].show();
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Method that shows the preview on the JPanels of the GUI.
	 * 
	 * @param imp Images Stack opened by ImageJ
	 */
	public void preview(ImagePlus imp) {
		previewPanel.removeAll();
		previewPanel.add(MyGUI.createLoading());

		previewPanel.revalidate();
		previewPanel.repaint();

		ImagePlus[] images = process(imp);

		previewPanel.removeAll();
		previewPanel.add(MyGUI.createPreviewWindow(images));

		previewPanel.revalidate();
		previewPanel.repaint();

	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * @param imp Images Stack opened by ImageJ
	 * @return Background, Sparse and Noise Images Stack (ImagePlus) in an array
	 */
	public ImagePlus[] process(ImagePlus imp) {
		System.gc(); // appel du garbage collector
		Runtime.getRuntime().gc();

		long startTime = System.nanoTime();

		/**
		 * originalImg It is matrix[rows*cols] where: cols = stackSize rows =
		 * width*height This means that each line represents one of the layers of the
		 * tif image
		 *
		 */

		// already transposed
		DenseMatrix64F originalImg = constructMatrix(imp.getStack(), dynamicRange);
		MyGUI.setProgressBar(3);

		/*
		 * SVD decomposition svdResult = ArrayList<X, Y, s> X = Unitary matrix having
		 * left singular vectors as columns. Y = Unitary matrix having right singular
		 * vectors as rows. s = Diagonal matrix with singular values.
		 */
		// ArrayList<SimpleMatrix> svdResult = svdDecomposition(originalImg);
		ArrayList<DenseMatrix64F> svdResult = randomizedSVD(originalImg, k);

		DenseMatrix64F X = svdResult.get(0);
		DenseMatrix64F Y = svdResult.get(1);
		DenseMatrix64F s = svdResult.get(2);

		// X = X * s
		DenseMatrix64F sX = new DenseMatrix64F(X.numRows, s.numCols);
		CommonOps.mult(X, s, sX);

		X = sX;

		System.gc(); // appel du arbage collector

		// L = X * Y
		DenseMatrix64F L = new DenseMatrix64F(X.numRows, Y.numCols);
		CommonOps.mult(X, Y, L);

		// S = originalImg - L
		DenseMatrix64F S = new DenseMatrix64F(originalImg.numRows, originalImg.numCols);
		CommonOps.sub(originalImg, L, S);

		// thresholding
		DenseMatrix64F thresholdS = threshold(S, tau, String.valueOf(mode));

		// error calculation
		int rankk = (int) Math.round((double) rank / k);
		DenseMatrix64F error = new DenseMatrix64F(rank * power, 1);

		// T = S - thresholdS
		DenseMatrix64F T = new DenseMatrix64F(S.numRows, S.numCols);
		CommonOps.sub(S, thresholdS, T);

		double normD = NormOps.normF(originalImg);
		double normT = NormOps.normF(T);

		error.set(0, normT / normD);

		int iii = 1;
		boolean stop = false;
		double alf;
		System.gc(); // appel du arbage collector

		int progress = 12;
		MyGUI.setProgressBar(progress);

		for (int i = 1; i < rankk + 1; i++) {
			/* Progress update */
			if (i == Math.floor(rankk / 2)) {
				progress = 43;
				MyGUI.setProgressBar(progress);
			}

			i = i - 1;
			int rrank = rank;
			alf = 0;
			double increment = 1;

			if (iii == power * (i - 2) + 1) {
				iii = iii + power;
			}

			System.gc(); // appel du garbage collector

			DenseMatrix64F LY = new DenseMatrix64F(L.numRows, Y.numRows);

			for (int j = 1; j < power + 1; j++) {

				/*
				 * X update
				 */

				// X = abs(L * transposedY)

				CommonOps.multTransB(L, Y, LY);
				X = LY;
				for (int k = 0; k < X.numCols; k++) {
					for (int l = 0; l < X.numRows; l++) {
						X.set(l, k, Math.abs(X.get(l, k)));
					}
				}

				/* Do a QR decomposition */
				QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(X.numRows, X.numCols);
				qr.decompose(X);
				qr.getQ(X, true);

				// X = QRFactorisation_Q(X, j);
				/*
				 * Y update
				 */
				// Y = transposedX * L
				CommonOps.multTransA(X, L, Y);

				// L = X * Y
				CommonOps.mult(X, Y, L);

				/*
				 * S update
				 */
				// T = originalImg - L
				CommonOps.sub(originalImg, L, T);
				// thresholding
				S = threshold(T, tau, String.valueOf(mode));

				// Error, stopping criteria
				// T = T - S
				CommonOps.sub(T, S, T);

				int ii = iii + j - 1;

				normT = NormOps.normF(T);
				error.set(ii, normT / normD);

				if (error.get(ii) < tol) {
					stop = true;
					break;
				}

				if (rrank != rank) {
					rank = rrank;
				}

				// Adjust alf
				double ratio = error.get(ii) / error.get(ii - 1);

				if (ratio >= 1.1) {
					increment = Math.max(0.1 * alf, 0.1 * increment);
					error.set(ii, error.get(ii - 1));
					alf = 0;
				} else if (ratio > 0.7) {
					increment = Math.max(increment, 0.25 * alf);
					alf = alf + increment;
				}

				/*
				 * Update of L
				 */
				// T = (1 + alf) * T
				CommonOps.scale(1 + alf, T, T);
				// L = L + T
				CommonOps.add(L, T, L);

				// Add corest AR
				if (j > 8) {
					double mean = mean(error, ii - 7, ii);
					if (mean > 0.92) {
						iii = ii;
						int YCol = Y.numCols;
						int XRow = X.numRows;
						if ((YCol - XRow) >= k) {
							CommonOps.extract(Y, 0, XRow - 1, 0, Y.numCols, Y, 0, 0);
						}
						break;
					}
				}

				System.gc(); // appel du garbage collector
			}

			if (stop) {
				break;
			}

//            	AR
			if (i + 1 < rankk) {
				Random r = new Random();
				DenseMatrix64F RR = new DenseMatrix64F(k, originalImg.numRows);
				for (int x = 0; x < k; x++) {
					for (int z = 0; z < originalImg.numRows; z++) {
						RR.set(x, z, r.nextGaussian());
						// RR.set(x, z, 1.0);
					}
				}
				// v = RR * L
				DenseMatrix64F v = new DenseMatrix64F(RR.numRows, L.numCols);
				CommonOps.mult(RR, L, v);
				// Y = combine(Y, v)
				DenseMatrix64F newY = new DenseMatrix64F(Y.numRows * 2, Y.numCols);
				for (int p = 0; p < Y.numRows; p++) {
					for (int q = 0; q < Y.numCols; q++) {
						newY.set(p, q, Y.get(p, q));
					}
				}
				for (int p = 0; p < v.numRows; p++) {
					for (int q = 0; q < v.numCols; q++) {
						newY.set(p + Y.numRows, q, v.get(p, q));
					}
				}
				Y = newY;
			}
			i++;

		}

		MyGUI.setProgressBar(75);

		// L = X * Y
		CommonOps.mult(X, Y, L);

		System.gc(); // appel du arbage collector

		/* Noise: G = originalImg - L - S */
		DenseMatrix64F G = new DenseMatrix64F(originalImg.numRows, originalImg.numCols);
		CommonOps.sub(originalImg, L, G);
		CommonOps.sub(G, S, G);

		MyGUI.setProgressBar(80);

		/* Construction des stack d'images */
		ImagePlus im = new ImagePlus(" Background Image ", constructImageStack(L, dynamicRange));
		ImagePlus im2 = new ImagePlus(" Sparse Image ", constructImageStack(S, dynamicRange));
		ImagePlus noise = new ImagePlus("Noise Image", constructImageStack(G, dynamicRange));

		long endTime = System.nanoTime();
		long duration = endTime - startTime;
		long durationInMilliseconds = duration / 1000000;
		double durationInSeconds = duration / 1000000000.0;
		System.out.println("Execution time in nanoseconds: " + duration);
		System.out.println("Execution time in milliseconds: " + durationInMilliseconds);
		System.out.println("Execution time in seconds: " + durationInSeconds);

		MyGUI.setProgressBar(100);

		return new ImagePlus[] { im, im2, noise };
	}

	/*-----------------------------------------------------------------------------------------------------------------------*/
	/**
	 * Thresholding mode
	 */
	private enum MODE {
		SOFT, HARD
	}
}
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------*/
