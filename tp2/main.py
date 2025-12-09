import os
import sys
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def find_roman_numeral(image_path, api_key=None):
    """
    Uses Gemini AI to find Roman numerals in an image.
    
    Args:
        image_path: Path to the input image file
        api_key: Google Gemini API key (optional, will use GEMINI_API_KEY from .env)
    
    Returns:
        The detected Roman numeral(s) as a string
    """
    # Get API key from parameter, .env file, or environment variable
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "API key not found. Please set GEMINI_API_KEY in your .env file "
            "or pass it as a parameter. Get your API key from: "
            "https://makersuite.google.com/app/apikey"
        )
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the model (using Gemini 1.5 Flash - supports vision and is faster)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Load and verify the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")
    
    # Create a prompt for Roman numeral detection
    prompt = """Analyze this image and identify any Roman numerals present. 
    Roman numerals use the letters I, V, X, L, C, D, M.
    
    Please respond with:
    1. The Roman numeral(s) you find in the image
    2. If multiple Roman numerals are present, list them all
    3. If no Roman numerals are found, respond with "No Roman numerals found"
    
    Be precise and only identify actual Roman numerals, not other text or symbols."""
    
    try:
        # Generate content using the image and prompt
        response = model.generate_content([prompt, img])
        
        # Extract the text response
        result = response.text.strip()
        return result
    
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")


def main():
    """Main function to handle command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [api_key]")
        print("\nExample:")
        print("  python main.py image.png")
        print("  python main.py image.png YOUR_API_KEY")
        print("\nNote: API key will be loaded from .env file (GEMINI_API_KEY)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        print(f"Analyzing image: {image_path}")
        print("Looking for Roman numerals...\n")
        
        result = find_roman_numeral(image_path, api_key)
        
        print("=" * 50)
        print("RESULT:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

