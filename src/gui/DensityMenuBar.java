package gui;

import java.awt.Menu;
import java.awt.MenuBar;
import java.awt.MenuItem;
import java.awt.MenuShortcut;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

public class DensityMenuBar extends MenuBar {
	/**
	 * For serialization.
	 */
	private static final long serialVersionUID = -8436059105842104698L;

	public static DensityMenuBar densityMenuBar = new DensityMenuBar();

	/********************* kmeans menu *********************/
	/**
	 * kMeans menu
	 */
	public Menu kMeansMenu;

	/**
	 * k means menu item
	 */
	public MenuItem kMeansMenuItem;

	/********************* density peaks menu *********************/
	/**
	 * dp menu
	 */
	public Menu dpMenu;

	/**
	 * dp menu item
	 */
	public MenuItem dpMenuItem;

	public DensityMenuBar() {
		super();

		// dp
		dpMenuItem = new MenuItem("DP clustering");
		dpMenuItem.addActionListener(new DpShower());
		dpMenuItem.setEnabled(true);

		dpMenu = new Menu("Density peaks");
		dpMenu.add(dpMenuItem);
		dpMenu.setEnabled(true);

		add(dpMenu);
	}// Of the constructor

	/**
	 *************************** 
	 * Show the load Density peaks dialog.
	 *************************** 
	 */
	private class DpShower implements ActionListener {
		public void actionPerformed(ActionEvent ae) {
			DpDialog.DpDialog.setVisible(true);
		}// Of actionPerformed
	}// Of DpShower
}
