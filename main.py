from Gui import SensorPlacementGUI
import matplotlib
import tkinter as tk

def main():
    matplotlib.use('TkAgg')
    root = tk.Tk()
    app = SensorPlacementGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
