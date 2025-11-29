
import sys
import os

# Add code directory to path
sys.path.append(os.path.join(os.getcwd(), 'code'))

try:
    import llm_module
    print("Successfully imported llm_module")
except Exception as e:
    print(f"Failed to import llm_module: {e}")
    import traceback
    traceback.print_exc()
