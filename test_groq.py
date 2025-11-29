import os
import sys

# Add the current directory to sys.path to ensure imports work
# Add the code directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'code'))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, relying on existing env vars")

from llm_module import LLM

def test_groq_generation():
    print("Testing Groq Integration...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    try:
        llm = LLM(backend="groq", model="openai/gpt-oss-20b")
        print("LLM initialized.")
        
        print("Generating response...")
        generator = llm.generate("What is 2 + 2? Answer in one word.")
        
        response = ""
        for token in generator:
            print(token, end="", flush=True)
            response += token
        print("\n")
        
        if "4" in response or "four" in response.lower():
            print("SUCCESS: Groq generation working correctly.")
        else:
            print(f"WARNING: Unexpected response: {response}")
            
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_groq_generation()
