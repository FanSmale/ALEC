package gui;

import java.awt.*;
import javax.swing.*;

/**
 * Project: The clustering comparison project. Author: <b>Fan Min</b>
 * minfanphd@163.com <br>
 * Last modify time: 2017/1/14.
 */
public class GUICommon extends Object {
	public static Frame mainFrame = null;

	public static JTabbedPane mainPane = null;

	/**
	 * Defaut font.
	 */
	public static final Font MY_FONT = new Font("Times New Romans", Font.PLAIN,
			12);

	/**
	 * Default color
	 */
	public static final Color MY_COLOR = Color.lightGray;

	/**
	 *************************** 
	 * Set the main frame. This can be done only once at the initialzing stage.
	 * 
	 * @param paramFrame
	 *            the main frame of the GUI.
	 *************************** 
	 */
	public static void setFrame(Frame paramFrame) throws Exception {
		if (mainFrame == null) {
			mainFrame = paramFrame;
		} else {
			throw new Exception("The main frame can be set only ONCE!");
		}// Of if
	}// Of setFrame

	/**
	 *************************** 
	 * Set the main pane. This can be done only once at the initialzing stage.
	 * 
	 * @param paramPane
	 *            the main pane of the GUI.
	 *************************** 
	 */
	public static void setPane(JTabbedPane paramPane) throws Exception {
		if (mainPane == null) {
			mainPane = paramPane;
		} else {
			throw new Exception("The main pane can be set only ONCE!");
		}// Of if
	}// Of setPAne

}// Of class GUICommon

