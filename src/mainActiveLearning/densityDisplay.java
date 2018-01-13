package mainActiveLearning;

import java.awt.Frame;

import gui.*;
//import coser.gui.guicommon.GUICommon;
//import coser.gui.menu.CoserMenuBar;

/**
 * The main entrance of the software.
 * 
 * @author Fan Min 2017.1.14.
 * 
 */
public class densityDisplay {
	/**
	 * The entrance method.
	 */
	public static void main(String args[]) {
		// A simple frame to contain dialogs.
		Frame mainFrame = new Frame();
		mainFrame
				.setTitle("Active learning through density cluster. http://www.fansmale.com, minfanphd@163.com");
		try {
			GUICommon.setFrame(mainFrame);
		} catch (Exception ee) {
			System.out.println(ee.toString());
			System.exit(0);
		}// Of try
		mainFrame.setMenuBar(DensityMenuBar.densityMenuBar);
		// 核心。ClusterMenuBar.clusterMenuBar是类加对象。
		// 用于生成各个munu
		// Basic settings of the frame
		mainFrame.setSize(800, 550);
		mainFrame.setLocation(50, 50);
		mainFrame.addWindowListener(ApplicationShutdown.applicationShutdown);
		// ApplicationShutdown.applicationShutdown，监听，有一个对象。
		// 用于对一系列事件进行反应。
		// 对所有事件进行响应。
		mainFrame.setBackground(GUICommon.MY_COLOR);
		mainFrame.setVisible(true);
	}// Of main
}
