package gui;

import java.io.*;
import java.awt.*;
import java.awt.event.*;

/**
 * Project: The cost-sensitive rough sets project.
 * <p>
 * Summary: Filename with the directory.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://grc.fjzs.edu.cn/>Lab of Granular Computing</a>,
 * Zhangzhou Normal University, Fujian 363000, China.<br>
 * Progress: OK. Can be enhanced further.<br>
 * Written time: March 09, 2011. <br>
 * Last modify time: March 09, 2011.
 */
public class FilenameField extends TextField implements ActionListener,
		FocusListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4796206438434365366L;

	/**
	 *************************** 
	 * No special initialization..
	 *************************** 
	 */
	public FilenameField() {
		super();
		setText("");
		addFocusListener(this);
	}// Of constructor

	/**
	 *************************** 
	 * No special initialization.
	 * 
	 * @param paraWidth
	 *            The width of the .
	 *************************** 
	 */
	public FilenameField(int paraWidth) {
		super(paraWidth);
		setText("");
		addFocusListener(this);
	}// Of constructor

	/**
	 *************************** 
	 * No special initialization.
	 * 
	 * @param paraWidth
	 *            The width of the .
	 * @param paraText
	 *            The given initial text
	 *************************** 
	 */
	public FilenameField(int paraWidth, String paraText) {
		super(paraWidth);
		setText(paraText);
		addFocusListener(this);
	}// Of constructor

	/**
	 *************************** 
	 * No special initialization.
	 * 
	 * @param paraWidth
	 *            The width of the .
	 * @param paraText
	 *            The given initial text
	 *************************** 
	 */
	public FilenameField(String paraText, int paraWidth) {
		super(paraWidth);
		setText(paraText);
		addFocusListener(this);
	}// Of constructor

	/**
	 ********************************** 
	 * Implement ActionListenter.
	 * 
	 * @param paraEvent
	 *            The event is unimportant.
	 ********************************** 
	 */
	public void actionPerformed(ActionEvent paraEvent) {
		FileDialog fileDialog = new FileDialog(GUICommon.mainFrame,
				"Select a file");
		// fileDialog.setSize(300, 300);
		fileDialog.setVisible(true);
		if (fileDialog.getDirectory() == null) {
			setText("");
			return;
		}
		String directoryName = fileDialog.getDirectory();
		// Use relative path if in the same area, Fan loves this one.

		setText(directoryName + fileDialog.getFile());
	}// Of actionPerformed

	/**
	 ********************************** 
	 * Implement FocusListenter.
	 * 
	 * @param paraEvent
	 *            The event is unimportant.
	 ********************************** 
	 */
	public void focusGained(FocusEvent paraEvent) {
	}// Of focusGained

	/**
	 ********************************** 
	 * Implement FocusListenter.
	 * 
	 * @param paraEvent
	 *            The event is unimportant.
	 ********************************** 
	 */
	public void focusLost(FocusEvent paraEvent) {
	}// Of focusLost

}// Of class FileSelecter

