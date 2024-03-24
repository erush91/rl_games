import os
import time

def append_df_to_csv_with_check(df, file_path, time_threshold=20):
    # Check file existence and age only if necessary
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > time_threshold:
            os.remove(file_path)
    
    # Use 'a+' mode to append; this creates the file if it doesn't exist
    with open(file_path, 'a+') as f:
        df.to_csv(f, mode='a', header=f.tell()==0, index=False)