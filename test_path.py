import os
import shutil
import tempfile
import contextlib
import cv2

# --- This is the function we are testing ---
@contextlib.contextmanager
def safe_video_path(path_with_spaces):
    safe_path = None
    print(f"\n[Inside safe_video_path] Received original path: '{path_with_spaces}'")
    try:
        # Step A: Create a temporary file placeholder
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            safe_path = tmp.name
        print(f"[Inside safe_video_path] Step A: Created temp placeholder at: '{safe_path}'")
        
        # Step B: Copy the original video to the temporary, safe path
        print(f"[Inside safe_video_path] Step B: Attempting to copy...")
        shutil.copy2(path_with_spaces, safe_path)
        print(f"[Inside safe_video_path] Step B: Copy successful.")
        
        # Yield the safe path for use outside
        yield safe_path
        
    finally:
        # Step D: Cleanup
        if safe_path and os.path.exists(safe_path):
            print(f"[Inside safe_video_path] Step D: Cleaning up and deleting temp file: '{safe_path}'")
            os.remove(safe_path)
        else:
            print(f"[Inside safe_video_path] Step D: No temp file to clean up.")

def run_test():
    print("--- Starting Final Debug Test ---")
    
    # --- !!! CRITICAL: CHANGE THIS FILENAME !!! ---
    # Use a real filename from your ORIGINAL dataset that has spaces.
    original_video_filename = 'SANI DIWEDI IVS.mp4' 
    # ------------------------------------------------
    
    original_path = f"../hb_meter_project_code/HB_new_dataset/{original_video_filename}"
    
    print(f"Testing with original relative path: '{original_path}'")
    
    # Pre-check: Does the original file exist?
    if not os.path.exists(original_path):
        print("\n\n>>> FATAL ERROR: The original source file does not exist at this path.")
        print(f">>> Checked path: {os.path.abspath(original_path)}")
        print(">>> Please verify the filename and relative path. The test cannot continue.")
        return

    print(">>> Pre-check PASSED: Original source file found.")

    try:
        # This is the main test of the copy-and-process logic
        with safe_video_path(original_path) as temp_path:
            print("\n[Main Test] Now inside the 'with' block.")
            print(f"[Main Test] The safe path to process is: '{temp_path}'")
            
            # Step C: Can OpenCV now read this temporary file?
            print("[Main Test] Step C: Attempting to open with OpenCV...")
            cap = cv2.VideoCapture(temp_path)
            is_opened = cap.isOpened()
            print(f"[Main Test] Step C: OpenCV open successful? -> {is_opened}")
            cap.release()
            if not is_opened:
                raise RuntimeError("OpenCV failed to open the temporary file!")

        # Check if cleanup worked
        if not os.path.exists(temp_path):
             print(f"\n>>> FINAL RESULT: SUCCESS! The process worked and the temp file was deleted.")
        else:
             print(f"\n>>> FINAL RESULT: PARTIAL SUCCESS. The process worked but cleanup failed.")

    except Exception as e:
        print(f"\n\n>>> FATAL ERROR DURING TEST: An exception occurred!")
        print(f">>> ERROR TYPE: {type(e).__name__}")
        print(f">>> ERROR MESSAGE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()