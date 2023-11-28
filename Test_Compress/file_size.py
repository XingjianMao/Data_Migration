
import os


def file_size(file_path):

    try:
        file_size = os.path.getsize(file_path)
        return (file_size)
        print(f"File Size in Bytes is {file_size}")
    except FileNotFoundError:
        print("File not found.")
    except OSError:
        print("OS error occurred.")
    