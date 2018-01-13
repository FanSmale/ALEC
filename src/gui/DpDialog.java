package gui;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Dialog;
import java.awt.GridLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileReader;

import javax.swing.JComboBox;

import algorithms.Density;

import gui.*;

public class DpDialog extends Dialog implements ActionListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1824087094625089819L;

	public static DpDialog DpDialog = new DpDialog();

	/**
	 * Select the arff file.
	 */
	private FilenameField arffFilenameField;

	/**
	 * The k value.
	 */
	private TextField kField;

	/**
	 * The k value.
	 */
	private TextField dcField;

	/**
	 * The t value.
	 */
	private TextField tField;

	/**
	 * The total value.
	 */
	private TextField totalField;

	/**
	 * The distance choice.
	 */
	private TextField distanceFiled;

	/**
	 * The ratioC value.
	 */
	private TextField ratioField;

	/**
	 *************************** 
	 * The only constructor.
	 *************************** 
	 */
	private DpDialog() {
		// This dialog is not module
		super(GUICommon.mainFrame, "Dp clustering", true);

		arffFilenameField = new FilenameField(30);
		arffFilenameField.setText("E:/arff/iris.arff");
		Button browseButton = new Button(" Browse ");
		browseButton.addActionListener(arffFilenameField);

		Panel sourceFilePanel = new Panel();
		sourceFilePanel.add(new Label("The arff file:"));
		sourceFilePanel.add(arffFilenameField);
		sourceFilePanel.add(browseButton);

		dcField = new TextField("0.1");
		kField = new TextField("2");
		ratioField = new TextField("0.01");
		distanceFiled = new TextField("1");
		tField = new TextField("2");
		totalField = new TextField("30000");

		Panel centerPanel = new Panel();
		centerPanel.setLayout(new GridLayout(1, 3));

		centerPanel.add(new Label("  k : "));
		centerPanel.add(kField);

		centerPanel.add(new Label("  dc: "));
		centerPanel.add(dcField);

		// centerPanel.add(new Label(" ratio : "));
		// centerPanel.add(ratioField);

		centerPanel.add(new Label(" distance : "));
		centerPanel.add(distanceFiled);

		centerPanel.add(new Label(" alpha : "));
		centerPanel.add(tField);

		centerPanel.add(new Label(" N : "));
		centerPanel.add(totalField);

		Button okButton = new Button(" OK ");
		okButton.addActionListener(this);
		DialogCloser dialogCloser = new DialogCloser(this);
		Button cancelButton = new Button(" Cancel ");
		cancelButton.addActionListener(dialogCloser);
		Button helpButton = new Button(" Help ");
		helpButton.setSize(20, 10);
		// helpButton.addActionListener(new HelpDialog("Test-cost distribution",
		// "coser/gui/dialog/tcnds/LoadTcNdshelp.txt"));
		Panel okPanel = new Panel();
		okPanel.add(okButton);
		okPanel.add(cancelButton);
		okPanel.add(helpButton);

		setLayout(new BorderLayout());
		add(BorderLayout.NORTH, sourceFilePanel);
		add(BorderLayout.CENTER, centerPanel);
		add(BorderLayout.SOUTH, okPanel);

		setBackground(GUICommon.MY_COLOR);
		setLocation(200, 200);
		setSize(420, 250);
		addWindowListener(dialogCloser);
		setVisible(false);
	}// Of constructor

	/**
	 *************************** 
	 * Read the arff file.
	 *************************** 
	 */
	public void actionPerformed(ActionEvent ae) {
		setVisible(false);
		ProgressDialog.progressDialog
				.setMessageAndShow("Please wait a few seconds. Computing ... \r\n");
		String resultString = "";
		try {
			FileReader fileReader = new FileReader(arffFilenameField.getText());
			Density tempDP = new Density(fileReader);
			fileReader.close();
			tempDP.setClassIndex(tempDP.numAttributes() - 1);
			tempDP.setDc(Double.parseDouble(dcField
					.getText()));

			tempDP.activeLearning(
					Integer.parseInt(kField.getText()),
					Integer.parseInt(tField.getText()),
					Integer.parseInt(totalField.getText()));
			
			
			resultString += "The Accuracy is: "
					+ tempDP.getPredictionAccuracy() + "\r\n";
			// resultString += "The numberTeach is: " + tempDP.numTeach +
			// "\r\n";

		} catch (Exception ee) {
			resultString += ee.getMessage();
		}// Of try

		ProgressDialog.progressDialog.setMessageAndShow(resultString);

	}// Of actionPerformed

}
