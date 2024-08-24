import tkinter as tk
from tkinter import filedialog

def choose_file():
    # Create a Tkinter root window, but keep it hidden
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog and store the selected file path
    file_path = filedialog.askopenfilename()

    # Print the file path
    print(f"Selected file: {file_path}")

# Call the function to open the file chooser and print the file path
choose_file()