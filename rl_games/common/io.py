import os
import time

def append_df_to_csv_with_check(df, file_path, time_threshold=20):
    """
    Appends a DataFrame to a CSV file, deleting the file first if it's older than the specified threshold.

    Parameters:
    - df: The DataFrame to append.
    - file_path: The path to the CSV file.
    - time_threshold: The age threshold in seconds for deleting the file. Defaults to 20 seconds.
    """
    # Check if the file exists and delete it if it's older than the threshold
    if os.path.exists(file_path):
        file_mod_time = os.path.getmtime(file_path)  # Get the file's last modification time
        current_time = time.time()  # Get the current time

        if (current_time - file_mod_time) > time_threshold:
            os.remove(file_path)  # Delete the file if it's older than the threshold

    # Append the DataFrame to the CSV file
    df.to_csv(file_path, mode='a', header=False, index=False)