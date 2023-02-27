package com.mycompany.imagej;

import ij.*;
import ij.gui.*;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

public class MyDialog extends GenericDialog implements ActionListener {

    private static MyDialog gd;

    public MyDialog(String title) {
        super(title);
    }
    @Override
    public void actionPerformed(ActionEvent e) {
        String action = e.getActionCommand();
        if (action.equals("  OK  ")) {

            // Process user's input
            Vector numberFields = gd.getNumericFields();
            Vector stringFields = gd.getStringFields();

            double tau = Double.parseDouble(((TextField) numberFields.get(0)).getText());
            String file = ((TextField) stringFields.get(0)).getText();

            GaussFiltering.setTau(tau);
            GaussFiltering.setFile(file);
            GaussFiltering.execute();
            // Do not dispose of the dialog
        } else if (action.equals("Cancel")) {
            gd.dispose();
        }
    }

    public static void createInterface()
    {
        String[] choice = {"soft", "hard"};

        gd = new MyDialog("Low Rank and Sparse tool");
        gd.addFileField("Input file: ", "");
        TextField tf = (TextField) gd.getStringFields().get(0);
        tf.setEditable(false);
        tf.setFocusable(false);
        gd.addNumericField("tau:", 7, 2);
        gd.addChoice("soft or hard thresholding", choice, choice[0]);
        gd.hideCancelButton();
        gd.showDialog();
    }

}
