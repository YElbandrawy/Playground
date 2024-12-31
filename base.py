import os
import platform
if platform.system() == 'Linux':
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
elif platform.system() == 'Windows':
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

class SensorOptimizer:
    def __init__(self, xyz_file, mode_files, target_sensors, modal_frequencies=np.array([
    1.4407, 2.2387, 2.3951, 2.9588, 3.5732, 4.1455,
    4.8339, 5.1074, 5.1398, 5.1825, 5.3577, 7.1458,
    7.3409, 7.4890, 8.8081, 9.6121, 9.9351, 10.022,
    10.183, 11.182
])):
        print("Initializing SensorOptimizer...")
        self.xyz_file = xyz_file
        self.mode_files = mode_files
        self.target_sensors = target_sensors
        self.modal_frequencies = modal_frequencies if modal_frequencies is not None else np.ones(len(mode_files))
        self.nodes = None
        self.Main_Mat = None
        self.POS = None
        self.COO = None
        self.Ed = None
  
    def read_coordinates(self):
            print("\nReading coordinates file...")
            try:
                df = pd.read_excel(self.xyz_file)
                coord_columns = ['X Location (mm)', 'Y Location (mm)', 'Z Location (mm)']
                self.nodes = df[coord_columns].values
                print(f"Successfully read {len(self.nodes)} nodes")
                return self.nodes
                
            except Exception as e:
                print(f"Error reading coordinate file: {str(e)}")
                raise
        
    def prepare_displacement_data(self):
        print("\nPreparing displacement data...")
        mode_data = []
        for i, file in enumerate(self.mode_files):
            try:
                print(f"Reading mode file {i+1}/{len(self.mode_files)}: {file}")
                df = pd.read_excel(file)
                data = df['Directional Deformation (mm)'].values
                mode_data.append(data)
            except Exception as e:
                print(f"Error reading mode file {file}: {str(e)}")
                raise
        
        self.Main_Mat = np.column_stack(mode_data)
        # Normalize the displacement data
        for i in range(self.Main_Mat.shape[1]):
            self.Main_Mat[:, i] = self.Main_Mat[:, i] / np.max(np.abs(self.Main_Mat[:, i]))
            
        print("Shape of Main_Mat:", self.Main_Mat.shape)
        self.POS = np.array([f"{i+1}" for i in range(len(self.nodes))])
        return self.Main_Mat

    def plot_nodes(self, nodes, title="Node Positions", selected_indices=None):
        print(f"\nPlotting {title}...")
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all nodes in light blue
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], 
                    c='lightblue', marker='o', alpha=0.5, label='All Nodes')
            
            # If we have selected nodes, plot them in red
            if selected_indices is not None:
                print(f"Plotting {len(selected_indices)} selected sensors")
                selected_nodes = nodes[selected_indices]
                # Increase size and make selected sensors more visible
                ax.scatter(selected_nodes[:, 0], selected_nodes[:, 1], selected_nodes[:, 2],
                        c='red', marker='o', s=200, alpha=1.0, label='Selected Sensors')
                
                # Print coordinates of selected sensors for verification
                for i, coord in enumerate(selected_nodes):
                    print(f"Selected sensor {i+1} at: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(title)
            if selected_indices is not None:
                ax.legend()
            
            plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved as {title.replace(' ', '_')}.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting nodes: {str(e)}")
            raise

    def effective_independence(self):
        print("\nRunning effective independence method...")
        M_Mat = self.Main_Mat.copy()
        n_dofs = M_Mat.shape[0]
        n_remove = n_dofs - self.target_sensors
        
        # Calculate initial FIM
        fim = M_Mat @ M_Mat.T
        eigenvals, eigenvects = eigh(fim)
        
        # Track indices of remaining nodes
        remaining_indices = np.arange(n_dofs)
        
        # Remove nodes in batches for better performance
        batch_size = max(100, n_remove // 10)
        
        while len(remaining_indices) > self.target_sensors:
            # Calculate contribution of each DOF
            Ed = np.sum(eigenvects**2, axis=1)
            
            # Get indices of nodes to remove in this batch
            n_to_remove = min(batch_size, len(remaining_indices) - self.target_sensors)
            remove_indices = np.argsort(Ed)[:n_to_remove]
            
            # Update remaining indices and matrices
            remaining_indices = np.delete(remaining_indices, remove_indices)
            M_Mat = np.delete(M_Mat, remove_indices, axis=0)
            
            # Recalculate FIM and eigenvectors
            fim = M_Mat @ M_Mat.T
            eigenvals, eigenvects = eigh(fim)
            
            print(f"Remaining nodes: {len(remaining_indices)}")
        
        # Calculate final contributions
        self.Ed = np.sum(eigenvects**2, axis=1)
        
        return remaining_indices, self.Ed
    
    def optimize_positions(self):
        print("\nStarting optimization process...")
        try:
            # Read and process input data
            self.read_coordinates()
            self.prepare_displacement_data()
            
            # Plot initial positions
            self.plot_nodes(self.nodes, "Initial Node Positions")
            
            # Run optimization
            selected_indices, contributions = self.effective_independence()
            
            # Store results
            results = {
                'POS': self.POS[selected_indices],
                'COO': self.nodes[selected_indices],
                'Ed': contributions
            }
            
            # Plot final positions with both selected and unselected nodes
            self.plot_nodes(self.nodes, "Selected Sensor Positions", selected_indices)
            
            # Save results
            self.save_results(results)
            
            print("\nOptimization completed successfully!")
            print("\nSelected sensor positions:")
            for i, (pos, coord) in enumerate(zip(results['POS'], results['COO']), 1):
                print(f"Sensor {i}: Node {pos} at coordinates ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
            
            return results
            
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            raise

    def calculate_dpr(self, mode_shapes):
        """
        Calculate Driving Point Residue for each DOF.
        
        Args:
            mode_shapes (np.ndarray): Mode shape matrix
        
        Returns:
            np.ndarray: DPR values for each DOF
        """
        dpr = np.zeros(mode_shapes.shape[0])
        for i in range(mode_shapes.shape[1]):
            # Square of mode shape amplitudes divided by modal frequency
            dpr += mode_shapes[:, i]**2 / self.modal_frequencies[i]
        return dpr

    def effective_independence_dpr(self):
        """
        Implement EFI-DPR method for sensor placement optimization.
        
        Returns:
            tuple: (selected indices, contribution measures)
        """
        print("\nRunning EFI-DPR method...")
        M_Mat = self.Main_Mat.copy()
        n_dofs = M_Mat.shape[0]
        
        # Calculate initial DPR
        dpr = self.calculate_dpr(M_Mat)
        
        # Normalize DPR values
        dpr_normalized = dpr / np.max(dpr)
        
        # Calculate initial FIM
        fim = M_Mat @ M_Mat.T
        eigenvals, eigenvects = eigh(fim)
        
        # Track indices of remaining nodes
        remaining_indices = np.arange(n_dofs)
        
        # Calculate initial EFI contribution
        Ed = np.sum(eigenvects**2, axis=1)
        Ed_normalized = Ed / np.max(Ed)
        
        # Calculate combined metric for all nodes
        combined_metric = Ed_normalized * dpr_normalized
        
        # Instead of iteratively removing nodes, directly select the top target_sensors nodes
        selected_indices = np.argsort(combined_metric)[-self.target_sensors:]
        
        # Calculate final contributions for selected nodes
        selected_M_Mat = M_Mat[selected_indices]
        selected_fim = selected_M_Mat @ selected_M_Mat.T
        _, selected_eigenvects = eigh(selected_fim)
        
        self.Ed = np.sum(selected_eigenvects**2, axis=1) * dpr_normalized[selected_indices]
        
        print(f"Selected {len(selected_indices)} sensor positions")
        return selected_indices, self.Ed

    def optimize_positions_dpr(self):
        """
        Execute the complete optimization process using EFI-DPR method.
        
        Returns:
            dict: Results containing selected positions, coordinates, and contributions
        """
        print("\nStarting EFI-DPR optimization process...")
        try:
            # Read and process input data
            self.read_coordinates()
            self.prepare_displacement_data()
            
            # Plot initial positions
            self.plot_nodes(self.nodes, "Initial Node Positions")
            
            # Run optimization with EFI-DPR
            selected_indices, contributions = self.effective_independence_dpr()
            
            # Store results
            results = {
                'POS': self.POS[selected_indices],
                'COO': self.nodes[selected_indices],
                'Ed': contributions
            }
            
            # Plot final positions with both selected and unselected nodes
            self.plot_nodes(self.nodes, "Selected Sensor Positions (EFI-DPR)", selected_indices)
            
            # Save results with EFI-DPR suffix
            self.save_results(results, suffix='_EFI_DPR')
            
            print("\nEFI-DPR optimization completed successfully!")
            print("\nSelected sensor positions (EFI-DPR method):")
            for i, (pos, coord) in enumerate(zip(results['POS'], results['COO']), 1):
                print(f"Sensor {i}: Node {pos} at coordinates ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
            
            return results
            
        except Exception as e:
            print(f"Error during EFI-DPR optimization: {str(e)}")
            raise
  
    def save_results(self, results, suffix=''):
            """
            Save optimization results to Excel files.
            
            Args:
                results (dict): Results to save
                suffix (str): Optional suffix for filenames
            """
            print("\nSaving results...")
            try:
                # Save sensor positions and contributions
                result_df = pd.DataFrame({
                    'Node': results['POS'],
                    'Contribution': results['Ed'],
                    'X (mm)': results['COO'][:, 0],
                    'Y (mm)': results['COO'][:, 1],
                    'Z (mm)': results['COO'][:, 2]
                })
                result_filename = f'resultBWPt{suffix}.xlsx'
                result_df.to_excel(result_filename, index=False)
                print(f"Saved results to {result_filename}")
                
                # Save reduced mode shape matrix
                mmat_filename = f'MMatBWPt{suffix}.xlsx'
                pd.DataFrame(self.Main_Mat).to_excel(mmat_filename, index=False)
                print(f"Saved mode shape matrix to {mmat_filename}")
                
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                raise

