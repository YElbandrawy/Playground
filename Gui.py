from optimizer import SensorOptimizer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

class SensorPlacementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimal Sensor Placement Tool")
        self.root.geometry("1200x800")
        
        # Variables
        self.n_modes = tk.StringVar(value="6")
        self.n_sensors = tk.StringVar(value="5")
        self.selected_files = []
        self.results_data = {}
        self.xyz_file = None
        self.mode_files = []
        self.optimizer = None
        self.current_view = tk.StringVar(value="")
        self.available_modes = []  # Store available modes
        self.mode_selection = []   # Store selected modes
        
        # Create main frames
        self.create_input_frame()
        self.create_method_frame()
        self.create_results_frame()
    
    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Mode selection frame
        mode_frame = ttk.LabelFrame(input_frame, text="Params", padding=5)
        mode_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        
        # Number of modes
        ttk.Label(mode_frame, text="N. Modes").grid(row=0, column=0, padx=5)
        mode_entry = ttk.Entry(mode_frame, textvariable=self.n_modes, width=10)
        mode_entry.grid(row=0, column=1, padx=5)
        mode_entry.bind('<Return>', self.update_mode_selection)
        
        # Number of sensors
        ttk.Label(mode_frame, text="N. Sensors").grid(row=0, column=2, padx=5)
        ttk.Entry(mode_frame, textvariable=self.n_sensors, width=10).grid(row=0, column=3, padx=5)
        
        # File input
        ttk.Label(input_frame, text="Input Files").grid(row=2, column=0, padx=5, pady=5)
        self.file_text = tk.Text(input_frame, height=5, width=50)
        self.file_text.grid(row=2, column=1, columnspan=3, pady=5)
        
        ttk.Button(input_frame, text="Load XYZ File", command=self.load_xyz_file).grid(row=2, column=4, padx=5)
        ttk.Button(input_frame, text="Load Mode Files", command=self.load_mode_files).grid(row=2, column=5, padx=5)

    def create_results_frame(self):
        results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        results_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create navigation list frame
        nav_frame = ttk.Frame(results_frame)
        nav_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ns")
        
        # Add navigation list
        self.nav_list = tk.Listbox(nav_frame, width=20, height=15)
        self.nav_list.pack(side=tk.LEFT, fill=tk.Y)
        nav_scrollbar = ttk.Scrollbar(nav_frame, orient="vertical", command=self.nav_list.yview)
        nav_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.nav_list.configure(yscrollcommand=nav_scrollbar.set)
        self.nav_list.bind('<<ListboxSelect>>', self.on_nav_select)
        
        # Create content frame
        content_frame = ttk.Frame(results_frame)
        content_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Results table with enhanced columns
        self.tree = ttk.Treeview(content_frame, columns=(
            "Node", 
            "X", 
            "Y", 
            "Z"
        ), show="headings")
        
        # Configure column headings
        self.tree.heading("Node", text="Node ID")
        self.tree.heading("X", text="X (mm)")
        self.tree.heading("Y", text="Y (mm)")
        self.tree.heading("Z", text="Z (mm)")
       
        
        # Configure column widths
        self.tree.column("Node", width=110)
        self.tree.column("X", width=110)
        self.tree.column("Y", width=110)
        self.tree.column("Z", width=110)
       
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Plot frame
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.grid(row=0, column=2, padx=8)

    def create_method_frame(self):
        method_frame = ttk.LabelFrame(self.root, text="Methods", padding=10)
        method_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Checkbuttons for methods
        self.efi_var = tk.BooleanVar()
        self.efi_dpr_var = tk.BooleanVar()
        
        ttk.Checkbutton(method_frame, text="EFI", variable=self.efi_var).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(method_frame, text="EFI-DPR", variable=self.efi_dpr_var).grid(row=0, column=1, padx=5)
        
        # Buttons
        ttk.Button(method_frame, text="select all", command=self.select_all).grid(row=0, column=3, padx=5, pady=10)
        ttk.Button(method_frame, text="Clear", command=self.clear_all).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(method_frame, text="OSP", command=self.run_osp).grid(row=1, column=1, padx=5, pady=10)
        ttk.Button(method_frame, text="EFI-genetic algo", command=self.run_efi_genetic).grid(row=1, column=2, padx=5, pady=10)
        ttk.Button(method_frame, text="EFI-DPR-genetic algo", command=self.run_efi_dpr_genetic).grid(row=1, column=3, padx=5, pady=10)

    def on_nav_select(self, event):
        selection = self.nav_list.curselection()
        if selection:
            selected_item = self.nav_list.get(selection[0])
            self.display_selected_result(selected_item)
    
    def load_xyz_file(self):
        self.xyz_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if self.xyz_file:
            self.update_file_text()
            
    def load_mode_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx")])
        if files:
            self.mode_files = list(files)
            
            # Group files by mode number
            x_files = [f for f in self.mode_files if 'DEFX' in f]
            y_files = [f for f in self.mode_files if 'DEFY' in f]
            z_files = [f for f in self.mode_files if 'DEFZ' in f]
            
            # Calculate maximum number of modes
            max_modes = min(len(x_files), len(y_files), len(z_files))
            
            if max_modes == 0:
                messagebox.showerror("Error", "No complete mode sets found")
                return
                
            self.update_file_text()
            self.n_modes.set(str(max_modes))
            self.update_mode_selection()
            messagebox.showinfo("Info", 
                f"Found {len(self.mode_files)} files representing {max_modes} modes.\n"
                f"Each mode consists of X, Y, and Z components.\n"
                f"Maximum number of modes you can analyze: {max_modes}")

    def update_file_text(self):
        self.file_text.delete(1.0, tk.END)
        if self.xyz_file:
            self.file_text.insert(tk.END, f"XYZ File: {os.path.basename(self.xyz_file)}\n")
        if self.mode_files:
            self.file_text.insert(tk.END, "Mode Files:\n")
            for file in self.mode_files:
                self.file_text.insert(tk.END, f"- {os.path.basename(file)}\n")
    
    def initialize_optimizer(self):
        try:
            n_sensors = int(self.n_sensors.get())
            n_modes = int(self.n_modes.get())
            
            if n_modes <= 0:
                messagebox.showerror("Error", "Please enter a valid number of modes")
                return False
            
            # Get files for all modes up to n_modes
            selected_files = []
            for mode_num in range(1, n_modes + 1):
                try:
                    x_file = next(f for f in self.mode_files if f'DEFX{mode_num}' in f)
                    y_file = next(f for f in self.mode_files if f'DEFY{mode_num}' in f)
                    z_file = next(f for f in self.mode_files if f'DEFZ{mode_num}' in f)
                    selected_files.extend([x_file, y_file, z_file])
                except StopIteration:
                    messagebox.showerror("Error", f"Could not find complete set of files for mode {mode_num}")
                    return False
                
            # Initialize optimizer with selected mode files
            self.optimizer = SensorOptimizer(
                xyz_file=self.xyz_file,
                mode_files=selected_files,
                target_sensors=n_sensors
            )
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize optimizer: {str(e)}")
            return False
            
    def run_osp(self):
        if not self.xyz_file or not self.mode_files:
            messagebox.showerror("Error", "Please load XYZ file and mode files first")
            return
            
        if not self.efi_var.get() and not self.efi_dpr_var.get():
            messagebox.showerror("Error", "Please select at least one method (EFI or EFI-DPR)")
            return
            
        try:
            if not self.initialize_optimizer():
                return
                
            # Clear previous results
            self.results_data.clear()
            self.nav_list.delete(0, tk.END)
            
            if self.efi_var.get():
                efi_results = self.optimizer.optimize_positions()
                self.results_data["EFI"] = efi_results
                self.nav_list.insert(tk.END, "EFI")
                
            if self.efi_dpr_var.get():
                efi_dpr_results = self.optimizer.optimize_positions_dpr()
                self.results_data["EFI-DPR"] = efi_dpr_results
                self.nav_list.insert(tk.END, "EFI-DPR")
                
            # Select first result by default
            if self.nav_list.size() > 0:
                self.nav_list.select_set(0)
                self.display_selected_result(self.nav_list.get(0))
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
        
    def run_efi_genetic(self):
        if not self.initialize_optimizer():
            return
        try:
            results = self.optimizer.optimize_positions_genetic(method='EFI')
            self.results_data["GA-EFI"] = results
            self.nav_list.insert(tk.END, "GA-EFI")
            self.nav_list.select_clear(0, tk.END)
            self.nav_list.select_set(tk.END)
            self.display_selected_result("GA-EFI")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def run_efi_dpr_genetic(self):
        if not self.initialize_optimizer():
            return
        try:
            results = self.optimizer.optimize_positions_genetic(method='EFI-DPR')
            self.results_data["GA-EFI-DPR"] = results
            self.nav_list.insert(tk.END, "GA-EFI-DPR")
            self.nav_list.select_clear(0, tk.END)
            self.nav_list.select_set(tk.END)
            self.display_selected_result("GA-EFI-DPR")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def clear_all(self):
        # Clear file selections
        self.xyz_file = None
        self.mode_files = []
        self.file_text.delete(1.0, tk.END)
        
        # Clear checkboxes
        self.efi_var.set(False)
        self.efi_dpr_var.set(False)
        
        # Clear results data
        self.results_data.clear()
        
        # Clear navigation list
        self.nav_list.delete(0, tk.END)
        
        # Clear results table
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Clear plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Reset optimizer
        self.optimizer = None
        
    def select_all(self):
        self.efi_var.set(True)
        self.efi_dpr_var.set(True)

    def display_selected_result(self, selected_item):
    # Clear existing displays
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        if selected_item not in self.results_data:
            messagebox.showwarning("Warning", "No data available for selected method.")
            return
            
        results = self.results_data[selected_item]
        
        try:
            # Update table with enhanced information including DOFs
            positions = results['COO']
            nodes = results['POS']
            contributions = results['Ed']
            
            # Sort sensors by nodeId value
            sorted_indices = np.argsort(nodes)[::-1]
            
            for rank, idx in enumerate(sorted_indices, 1):
                coord = positions[idx]
                node = nodes[idx]
                contrib = contributions[idx]
                
                # Calculate DOFs based on selected modes
                selected_modes = [i+1 for i in range(len(self.available_modes)) 
                                if self.mode_listbox.selection_includes(i)]
                dof_str = f"Modes {','.join(map(str, selected_modes))}"

                self.tree.insert('', 'end', values=(
                    f"{node}",         # Node ID
                    f"{coord[0]:.2f}", # X coordinate
                    f"{coord[1]:.2f}", # Y coordinate
                    f"{coord[2]:.2f}"  # Z coordinate
                ))
            
            # Rest of the visualization code remains the same...
            # [Previous 3D visualization code goes here]
            # Create 3D visualization
            fig = plt.Figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all nodes if available
            if self.optimizer and self.optimizer.nodes is not None:
                all_nodes = self.optimizer.nodes
                ax.scatter(all_nodes[:, 0], all_nodes[:, 1], all_nodes[:, 2],
                          c='lightblue', marker='o', alpha=0.2, s=20,label='Available Nodes')
            
            # Plot selected sensors
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            color='red',s=100, alpha=0.5, marker='o', label='Selected Sensors')
 
            # Add node labels for selected sensors
            for i, (coord, node) in enumerate(zip(positions, nodes)):
                ax.text(coord[0], coord[1], coord[2], f' {node}',
                       fontsize=8, ha='left', va='bottom')
            
            # Set labels and title
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f'Sensor Placement - {selected_item}')
            
            # Add legend
            ax.legend()
            
            # Adjust view
            ax.view_init(elev=30, azim=45)
            
            # Add interaction capabilities
            def on_click(event):
                if event.inaxes == ax:
                    ax.view_init(elev=ax.elev, azim=ax.azim)
                    canvas.draw()
            
            # Create toolbar for basic matplotlib interactions
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add navigation toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
            # Connect click event
            canvas.mpl_connect('button_press_event', on_click)
            
            # Add summary information
            summary_frame = ttk.Frame(self.plot_frame)
            summary_frame.pack(fill=tk.X, pady=5)
            
            summary_text = (
                f"Method: {selected_item}\n"
                f"Total Sensors: {len(positions)}\n"
            )
            
            summary_label = ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT)
            summary_label.pack(padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying results: {str(e)}")
            print(f"Error in display_selected_result: {str(e)}")
            import traceback
            traceback.print_exc()
              
    def update_mode_selection(self, event=None):
        try:
            n_modes = int(self.n_modes.get())
            max_modes = len([f for f in self.mode_files if 'DEFX' in f])  # Count X files only
            
            if n_modes > max_modes:
                messagebox.showerror("Error", 
                    f"Maximum number of modes cannot exceed {max_modes}\n"
                    f"(based on complete XYZ mode sets)")
                self.n_modes.set(str(max_modes))
                n_modes = max_modes
            
            # Clear and update the listbox
            self.mode_listbox.delete(0, tk.END)
            self.available_modes = [f"Mode {i+1} (X,Y,Z)" for i in range(n_modes)]
            for mode in self.available_modes:
                self.mode_listbox.insert(tk.END, mode)
            
            # Select all modes automatically
            self.mode_listbox.select_set(0, n_modes - 1)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of modes")
    