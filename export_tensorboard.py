import argparse
import os
from tbparse import SummaryReader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

def find_latest_run(base_log_dir):
    """Finds the latest run directory in the base log directory."""
    print(f"Searching for subdirectories in: {os.path.abspath(base_log_dir)}")
    try:
        all_entries = os.listdir(base_log_dir)
    except FileNotFoundError:
        return None

    subdirs = [os.path.join(base_log_dir, d) for d in all_entries if os.path.isdir(os.path.join(base_log_dir, d))]

    if not subdirs:
        return None
    
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def get_tensorboard_data(log_dir):
    """Reads all scalar data from a TensorBoard log directory."""
    print(f"Reading logs from: {log_dir}")
    try:
        reader = SummaryReader(log_dir)
        # The .scalars property returns a single DataFrame with all scalar data
        all_scalars_df = reader.scalars
    except Exception as e:
        print(f"Error: Could not read TensorBoard logs from '{log_dir}'.")
        print(f"Please ensure this is the correct directory and it contains event files.")
        print(f"Details: {e}")
        return None

    if all_scalars_df.empty:
        print("No scalar data (graphs) found in the specified log directory.")
        return None
    
    # Group the single DataFrame by 'tag' to create a dictionary of DataFrames
    data_dict = {tag: group_df for tag, group_df in all_scalars_df.groupby('tag')}
        
    print(f"Found {len(data_dict)} graphs to export.")
    return data_dict

def create_pdf_report(data, output_filename):
    """Creates a multi-page PDF report from a dictionary of TensorBoard data."""
    try:
        with PdfPages(output_filename) as pdf:
            # --- First Page: Table of Contents ---
            tags = list(data.keys())
            
            fig, ax = plt.subplots(figsize=(8.5, 11)) # Portrait for TOC
            # Adjust title position and font size for a more compact look
            fig.suptitle('TensorBoard Export - Table of Contents', fontsize=16, y=0.97)
            ax.axis('off')

            toc_text = "\n".join([f"- {tag}" for tag in tags])
            
            # Smaller font size to fit more items on one page
            ax.text(0.05, 0.95, toc_text, transform=ax.transAxes, fontsize=8, 
                    verticalalignment='top', family='monospace')
            
            pdf.savefig(fig)
            plt.close()

            # --- Subsequent Pages: Graphs ---
            for i, (tag, df) in enumerate(data.items()):
                print(f"  - Plotting graph {i+1}/{len(data)}: {tag}")
                fig, ax = plt.subplots(figsize=(11, 8.5)) # Landscape for graphs
                
                ax.plot(df['step'], df['value'])
                
                ax.set_title(tag, fontsize=12)
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.grid(True)
                
                fig.tight_layout(pad=3.0)
                pdf.savefig(fig)
                plt.close(fig)

        print(f"\nSuccessfully created PDF report: {output_filename}")

    except Exception as e:
        print(f"\nError: Failed to create PDF report.")
        print(f"Details: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar data to a multi-page PDF report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help="Path to the directory containing TensorBoard event files.\nIf not provided, the script will search for the latest run in './logs/co_rl/'."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='tensorboard_report.pdf',
        help="Path to save the output PDF file (default: 'tensorboard_report.pdf')."
    )

    args = parser.parse_args()
    
    log_directory = args.log_dir
    if log_directory is None:
        base_log_dir = "logs/co_rl"
        print(f"No log directory provided. Searching for the latest run in '{base_log_dir}'...")
        log_directory = find_latest_run(base_log_dir)
        if log_directory is None:
            print(f"Error: No subdirectories found in '{base_log_dir}'. Please specify a --log_dir.")
            exit()
        print(f"Found latest run: {log_directory}")

    # Get the data from TensorBoard logs
    tb_data = get_tensorboard_data(log_directory)
    
    # If data was found, create the PDF report
    if tb_data:
        create_pdf_report(tb_data, args.output_file)