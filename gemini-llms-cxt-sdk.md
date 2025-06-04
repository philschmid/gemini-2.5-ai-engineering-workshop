<project title="Google Gemini SDK Comprehensive Guide" summary="A complete guide to the Google Gemini Python SDK, detailing its features, syntax, and best practices for building AI applications. This context is designed for an LLM to understand and effectively utilize the Gemini API.">
<!-- SDK Initialization and Core Concepts -->
The Google Gemini Python SDK enables interaction with Google's Gemini models.

**SDK Installation:**
The SDK is installed using pip:
```python
# pip install -U -q google-genai
```
Ensure this package is available in the execution environment.

**Client Initialization:**
The primary interface to the API is the `genai.Client` class. It requires an API key for authentication.
```python
from google import genai
import os

# The API key is typically provided via an environment variable.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    # In a real application, this would be an error or a configuration issue.
    # For an LLM, this indicates the necessity of the key.
    print("GEMINI_API_KEY environment variable not set.")
    # It's crucial for the LLM to know this key MUST be provided for the client to work.
    # raise ValueError("API Key not found.")

client = genai.Client(api_key=GEMINI_API_KEY)
```

**Available Models:**
Various Gemini models are available, each suited for different tasks and capabilities (e.g., speed, context window size, multimodality). Model IDs are specified in API calls.
Examples include:
- `gemini-2.0-flash`
- `gemini-2.5-flash-preview-05-20` (multimodal, used extensively in examples)
- `gemini-2.5-pro-preview-05-06` (best reasoning model, paid)
- `gemini-2.0-flash-preview-image-generation` (for image generation)
- `gemini-2.5-flash-preview-tts` (for text-to-speech)
Model availability and IDs can change; refer to official Google AI documentation for the most current list. API calls often use the format `models/MODEL_ID_STRING`. A typical `MODEL_ID` used in examples for general multimodal tasks:
```python
MODEL_ID = "gemini-2.5-flash-preview-05-20"
```

**Token Management:**
API usage is often measured in tokens. Understanding token counts is important for managing context limits and potential costs.
- **Counting Tokens:** Use `client.models.count_tokens(model="models/MODEL_ID", contents=...)` to get the token count for a given input.
  ```python
  prompt_content = "This is a sample text to count tokens."
  # Assuming MODEL_ID is defined as above.
  token_count_response = client.models.count_tokens(model=f"models/{MODEL_ID}", contents=prompt_content)
  print(f"Tokens: {token_count_response.total_tokens}")
  ```
- **Usage Metadata:** After a generation call, `response.usage_metadata` provides details on `prompt_token_count`, `candidates_token_count`, and `total_token_count`. Some responses may include `thoughts_token_count`.

**Error Handling:**
API calls can raise exceptions (e.g., from `google.api_core.exceptions` like `InvalidArgument`, `PermissionDenied`, `ResourceExhausted`). Robust applications should handle these.

<docs>
    <doc title="Text Generation and Chat" desc="Covers single-turn text generation, streaming for interactive experiences, multi-turn chat conversations, system instructions, and generation configuration.">
# Text Generation and Chat Capabilities

## Basic Text Generation
The `client.models.generate_content()` method is used for single-turn, non-streaming text generation.
```python
from google.genai import types # For configuration objects

# Assume client and MODEL_ID are defined (e.g., MODEL_ID = "gemini-2.5-flash-preview-05-20").
simple_prompt = "What is the speed of light?"
response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=simple_prompt
)
print(response.text) # Access the generated text
```
The `contents` parameter can be a single string or an iterable of strings/`types.Part` objects.

## Streaming Responses
For applications requiring incremental delivery of text (e.g., chatbots, long-form content generation), use `client.models.generate_content_stream()`.
```python
streaming_prompt = "Tell me a long story about a space explorer."
response_stream = client.models.generate_content_stream(
    model=f"models/{MODEL_ID}",
    contents=streaming_prompt
)
for chunk in response_stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
```

## Chat (Multi-turn Conversations)
The SDK provides `client.chats.create()` to manage conversational context.
```python
chat_session = client.chats.create(model=f"models/{MODEL_ID}")

user_message1 = "Hello, who are you?"
response1 = chat_session.send_message(message=user_message1)
print(f"User: {user_message1}\nModel: {response1.text}\n")

user_message2 = "What can you do?"
response2 = chat_session.send_message(message=user_message2)
print(f"User: {user_message2}\nModel: {response2.text}\n")

# Conversation history is accessible via chat_session.history or chat_session.get_history()
for message_content in chat_session.history:
    print(f"{message_content.role.capitalize()}: {message_content.parts[0].text}")
```

## System Instructions
System instructions guide the model's behavior, personality, or output format. They are applied via `types.GenerateContentConfig`.
```python
system_instruction_text = "You are a helpful assistant that speaks like a pirate."
pirate_config = types.GenerateContentConfig(system_instruction=system_instruction_text)

# For single generation:
pirate_response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents="What's the weather today?",
    config=pirate_config
)
print(pirate_response.text)

# For chat sessions:
pirate_chat = client.chats.create(
    model=f"models/{MODEL_ID}",
    config=pirate_config
)
chat_response = pirate_chat.send_message("Suggest a good book to read.")
print(chat_response.text)
```

## Generation Configuration
Fine-tune model output using `types.GenerateContentConfig` or a dictionary passed to the `config` parameter of generation methods.
Key parameters include:
- `temperature` (float, 0.0-2.0): Controls randomness. Lower values are more deterministic.
- `max_output_tokens` (int): Limits the length of the generated response.
- `top_p` (float, 0.0-1.0): Nucleus sampling; controls diversity.
- `top_k` (int): Considers the `top_k` most likely tokens at each step.
- `stop_sequences` (List[str]): Sequences where the model should stop generating.
```python
generation_settings = {
    "temperature": 0.3,
    "max_output_tokens": 100,
    "top_p": 0.9,
    "top_k": 40,
}
configured_response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents="Write a short poem about the moon.",
    config=generation_settings
)
print(configured_response.text)
```
    </doc>
    <doc title="Multimodal Capabilities and File Handling" desc="Working with images, audio, video, and documents (PDFs). Covers uploading files using the File API and constructing multimodal prompts. Also includes code understanding, image generation, and text-to-speech.">
# Multimodal Capabilities and File Handling

Gemini models can process various content types in a single prompt. This is achieved by providing an iterable of `types.Part` objects to the `contents` parameter.

## File API for Large or Reusable Content
The File API (`client.files`) allows uploading files (images, audio, video, PDFs) for use in prompts. Uploaded files are stored for 48 hours (max 2GB per file, 20GB per project).
```python
# Ensure 'my_document.pdf' exists in the specified path for this to run.
file_to_upload_path = "my_document.pdf" # Placeholder path
if os.path.exists(file_to_upload_path):
    try:
        uploaded_file_obj = client.files.upload(path=file_to_upload_path, display_name="My Document")
        print(f"File uploaded: {uploaded_file_obj.name}, URI: {uploaded_file_obj.uri}")

        # Use in a prompt:
        prompt_with_file = [
            "Summarize this document:",
            uploaded_file_obj # Pass the File object directly
        ]
        response = client.models.generate_content(model=f"models/{MODEL_ID}", contents=prompt_with_file)
        print(response.text)

        # Delete the file when no longer needed (optional, auto-expires)
        # client.files.delete(name=uploaded_file_obj.name)
    except Exception as e:
        print(f"File API error: {e}")
else:
    print(f"File not found: {file_to_upload_path}. Skipping File API example part.")
```

## Image Understanding
Process images provided as raw bytes or via the File API.
```python
from io import BytesIO
import requests # For fetching image from URL

# Example: Image from bytes (e.g., fetched from a URL)
image_url = "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
try:
    img_response = requests.get(image_url)
    img_response.raise_for_status()
    image_bytes_data = img_response.content
    image_part_from_bytes = types.Part.from_bytes(data=image_bytes_data, mime_type="image/jpeg")

    image_prompt_parts = ["What is depicted in this image?", image_part_from_bytes]
    response_img_desc = client.models.generate_content(
        model=f"models/{MODEL_ID}", # Use a multimodal model
        contents=image_prompt_parts
    )
    print(response_img_desc.text)
except Exception as e:
    print(f"Image understanding error: {e}")

# For multiple images, include multiple image_part objects in `contents`.
```

## Audio Understanding
Process audio files (MP3, WAV, FLAC, AAC etc.) typically uploaded via the File API.
```python
# Assume 'sample_audio.mp3' exists for this to run.
audio_file_path = "sample_audio.mp3" # Placeholder path
if os.path.exists(audio_file_path):
    try:
        uploaded_audio_file = client.files.upload(path=audio_file_path)
        audio_part_from_uri = types.Part.from_uri(
            uri=uploaded_audio_file.uri,
            mime_type=uploaded_audio_file.mime_type # e.g., "audio/mpeg"
        )
        audio_prompt = ["Transcribe this audio.", audio_part_from_uri]
        response_audio_transcription = client.models.generate_content(
            model=f"models/{MODEL_ID}", contents=audio_prompt
        )
        print(response_audio_transcription.text)
    except Exception as e:
        print(f"Audio processing error: {e}")
else:
    print(f"Audio file not found: {audio_file_path}. Skipping audio example part.")
```

## Video Understanding
Process video files. Large videos are uploaded via File API and may require processing time.
```python
from time import sleep
# Assume 'sample_video.mp4' exists for this to run.
video_file_path_example = "sample_video.mp4" # Placeholder path
if os.path.exists(video_file_path_example):
    try:
        video_file_upload = client.files.upload(path=video_file_path_example)
        # Wait for processing
        while video_file_upload.state.name == "PROCESSING":
            print("Video processing...")
            sleep(10) # Check every 10 seconds; adjust based on expected processing time.
            video_file_upload = client.files.get(name=video_file_upload.name)
        if video_file_upload.state.name == "ACTIVE":
            video_part = types.Part.from_uri(uri=video_file_upload.uri, mime_type=video_file_upload.mime_type)
            video_prompt = ["Summarize this video.", video_part]
            response_video_summary = client.models.generate_content(model=f"models/{MODEL_ID}", contents=video_prompt)
            print(response_video_summary.text)
        else:
            print(f"Video processing failed: {video_file_upload.state.name}, Error: {video_file_upload.error}")
    except Exception as e:
        print(f"Video processing error: {e}")
else:
    print(f"Video file not found: {video_file_path_example}. Skipping video example part.")

# YouTube Video Analysis:
youtube_url = "https://www.youtube.com/watch?v=your_video_id" # Replace with actual ID
# The SDK handles ingestion. For this to work, the video must be public or unlisted and accessible.
youtube_part = types.Part(file_data=types.FileData(file_uri=youtube_url))
# Or, more explicitly (and often preferred if MIME type is certain):
# youtube_part = types.Part.from_uri(uri=youtube_url, mime_type="video/youtube")
youtube_prompt = ["What is this YouTube video about?", youtube_part]
try:
    response_youtube = client.models.generate_content(model=f"models/{MODEL_ID}", contents=youtube_prompt)
    print(response_youtube.text)
except Exception as e:
    print(f"YouTube analysis error: {e}")
```

## PDF and Document Understanding
Extract information from PDF and other document formats using the File API.
```python
# Assume 'your_invoice.pdf' exists for this to run.
pdf_file_to_analyze = "your_invoice.pdf" # Placeholder path
if os.path.exists(pdf_file_to_analyze):
    try:
        uploaded_pdf = client.files.upload(path=pdf_file_to_analyze)
        pdf_part = types.Part.from_uri(uri=uploaded_pdf.uri, mime_type=uploaded_pdf.mime_type)
        pdf_prompt = ["Extract the total amount due from this invoice.", pdf_part]
        response_pdf_extract = client.models.generate_content(model=f"models/{MODEL_ID}", contents=pdf_prompt)
        print(response_pdf_extract.text)
    except Exception as e:
        print(f"PDF processing error: {e}")
else:
    print(f"PDF file not found: {pdf_file_to_analyze}. Skipping PDF example part.")
```

## Code Understanding
Gemini can understand and discuss code provided as text within the `contents`.
```python
python_code_snippet = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""
code_prompt = f"Explain this Python code:\n\n{python_code_snippet}"
response_code_exp = client.models.generate_content(model=f"models/{MODEL_ID}", contents=code_prompt)
print(response_code_exp.text)
```

## Image Generation
Generate images from text prompts using specific image generation models.
```python
from PIL import Image # For handling image data

IMAGE_GEN_MODEL_ID = "gemini-2.0-flash-preview-image-generation" # Example model ID from workshop
image_gen_prompt = "A surreal painting of a whale flying through a nebula."
try:
    img_gen_response = client.models.generate_content(
        model=f"models/{IMAGE_GEN_MODEL_ID}",
        contents=image_gen_prompt,
        config=types.GenerateContentConfig(
          response_modalities=['IMAGE', 'TEXT'] # Request image and optional text
        )
    )
    # Access the generated image data
    generated_image_found = False
    for part in img_gen_response.candidates[0].content.parts:
        if part.text:
            print(f"Image Gen Text Response: {part.text}")
        elif part.inline_data and part.inline_data.mime_type.startswith('image/'):
            generated_image_bytes = part.inline_data.data
            image = Image.open(BytesIO(generated_image_bytes))
            image.save("generated_gemini_image.png")
            print("Image generated and saved as generated_gemini_image.png.")
            generated_image_found = True
            break # Assuming one image per prompt here
    if not generated_image_found:
        print("No image data found in the response.")
except Exception as e:
    print(f"Image generation error: {e}. Ensure model '{IMAGE_GEN_MODEL_ID}' is available and supports image generation.")
```

## Text-to-Speech (TTS)
Convert text into speech audio using specific TTS models.
```python
import numpy as np
import soundfile as sf # For saving audio file

TTS_MODEL_ID = "gemini-2.5-flash-preview-tts" # Example model ID from workshop
text_for_speech = "Hello from Gemini. This is an audio test."
try:
    tts_response = client.models.generate_content(
       model=f"models/{TTS_MODEL_ID}",
       contents=text_for_speech,
       config=types.GenerateContentConfig(
          response_modalities=["AUDIO"],
          speech_config=types.SpeechConfig(
             voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore') # Example voice
             )
          )
       )
    )
    # Access generated audio data
    if tts_response.candidates[0].content.parts[0].inline_data:
        audio_bytes = tts_response.candidates[0].content.parts[0].inline_data.data
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        sf.write("gemini_speech.wav", audio_array, 24000) # Sample rate 24kHz is common
        print("Speech audio generated and saved as gemini_speech.wav.")
    else:
        print("No audio data found in TTS response.")
except Exception as e:
    print(f"TTS error: {e}. Ensure model '{TTS_MODEL_ID}' is available and supports TTS.")
```
    </doc>
    <doc title="Structured Outputs, Function Calling, and Native Tools" desc="Techniques for obtaining structured JSON responses, integrating external functions/APIs (function calling), and utilizing built-in native tools like Google Search, URL Context, and Code Execution.">
# Structured Outputs, Function Calling, and Native Tools

## Structured Outputs
Constrain model responses to a specific JSON schema using Pydantic models.
```python
from pydantic import BaseModel
from typing import List

class BookInfo(BaseModel):
    title: str
    author: str
    year: int

struct_prompt = "Provide information for the book '1984' by George Orwell."
try:
    struct_response = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=struct_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BookInfo # Pass the Pydantic model class
        )
    )
    if hasattr(struct_response, 'parsed') and struct_response.parsed:
        book_data: BookInfo = struct_response.parsed
        print(f"Title: {book_data.title}, Author: {book_data.author}, Year: {book_data.year}")
    else:
        print(f"Could not parse structured output. Raw response: {struct_response.text}")
except Exception as e:
    print(f"Structured output error: {e}")
```

## Function Calling
Enable models to interact with external systems by defining functions the model can request to call.

### Manual Function Calling Flow
Involves explicit steps to handle the model's request to call a function and provide its result.
```python
# 1. Define your Python function
def get_stock_price(ticker: str) -> dict:
    """Gets the mock stock price for a ticker."""
    print(f"SDK FUNCTION: get_stock_price(ticker='{ticker}') called.")
    if ticker.upper() == "GOOGL": return {"ticker": "GOOGL", "price": 170.00, "currency": "USD"}
    return {"ticker": ticker, "price": "unknown", "currency": "USD"}

# 2. Declare the function for the model
stock_price_declaration = types.FunctionDeclaration(
    name="get_stock_price",
    description="Fetches the current stock price for a given ticker symbol.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"ticker": types.Schema(type=types.Type.STRING, description="The stock ticker symbol")},
        required=["ticker"]
    )
)
tools_manual_fc = [types.Tool(function_declarations=[stock_price_declaration])]

# 3. Initial request and model's function call
current_contents = [types.Content(role="user", parts=[types.Part(text="What's the stock price for GOOGL?")])]
try:
    response_turn1 = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=current_contents,
        config=types.GenerateContentConfig(tools=tools_manual_fc)
    )
    model_content_turn1 = response_turn1.candidates[0].content
    current_contents.append(model_content_turn1) # Add model's response (which might be a function call) to history

    # 4. Check for function call and execute
    if model_content_turn1.parts and model_content_turn1.parts[0].function_call:
        fc = model_content_turn1.parts[0].function_call
        print(f"Model requests call to: {fc.name}, Args: {dict(fc.args)}")
        if fc.name == "get_stock_price":
            api_result = get_stock_price(ticker=fc.args['ticker'])
            # 5. Send function response back to model
            function_response_part = types.Part.from_function_response(name=fc.name, response={"result": api_result})
            # Gemini API expects the function response to be part of a "user" role message in the history.
            current_contents.append(types.Content(role="user", parts=[function_response_part]))
            
            response_turn2 = client.models.generate_content(
                model=f"models/{MODEL_ID}",
                contents=current_contents, # Send updated history
                config=types.GenerateContentConfig(tools=tools_manual_fc) # Tools might be needed again
            )
            print(f"Final Model Response: {response_turn2.text}")
    else:
        print(f"Model answered directly: {response_turn1.text}")
except Exception as e:
    print(f"Manual function calling error: {e}")
```

### Automatic Function Calling (Python SDK)
The SDK can manage the function calling flow automatically if Python functions (with type hints and docstrings) are provided.
```python
# Define functions as before (e.g., get_stock_price with type hints and docstring)
auto_fc_tools = [get_stock_price] # Pass the callable Python functions directly
auto_fc_config = types.GenerateContentConfig(tools=auto_fc_tools)

try:
    auto_fc_response = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents="What is the price of GOOGL stock?",
        config=auto_fc_config
    )
    print(f"Automatic FC Response: {auto_fc_response.text}")
except Exception as e:
    print(f"Automatic function calling error: {e}")
```

## Native Tools
Gemini offers built-in tools for common tasks.

### Google Search Tool
Grounds responses with information from Google Search.
```python
search_tool = types.Tool(google_search=types.GoogleSearch())
search_prompt = "Who won the latest Formula 1 race?"
try:
    search_response = client.models.generate_content(
        model=f"models/{MODEL_ID}", # Or a model specifically good with search
        contents=search_prompt,
        config=types.GenerateContentConfig(tools=[search_tool])
    )
    print(f"Search-grounded response: {search_response.text}")
    # Citations might be available in search_response.candidates[0].citation_metadata
    if search_response.candidates[0].citation_metadata:
        print(f"Citations: {search_response.candidates[0].citation_metadata.citation_sources}")
except Exception as e:
    print(f"Google Search tool error: {e}")
```

### URL Context Tool
Allows the model to use content from specified URLs.
```python
url_tool = types.Tool(url_context=types.UrlContext())
url_prompt = "Summarize the main points of the article at https://blog.google/technology/ai/google-gemini-ai/"
try:
    url_response = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=url_prompt,
        config=types.GenerateContentConfig(tools=[url_tool])
    )
    print(f"URL context response: {url_response.text}")
except Exception as e:
    print(f"URL Context tool error: {e}")
```

### Code Execution Tool
Enables the model to generate and execute Python code.
```python
# For displaying images in environments like Jupyter:
# from IPython.display import Image as IPImage, Markdown

code_exec_tool = types.Tool(code_execution={}) # Enable by passing an empty dict
code_exec_prompt = "Calculate the square root of 12345 and then add 100 to it. Show the result. Also, plot sin(x) from 0 to 2*pi."
try:
    code_exec_response = client.models.generate_content(
        model=f"models/{MODEL_ID}", # A model supporting code execution
        contents=code_exec_prompt,
        config=types.GenerateContentConfig(tools=[code_exec_tool])
    )
    # Iterate through parts to find text, executable code, or results (like images)
    print("\nCode Execution Tool Output:")
    for part in code_exec_response.candidates[0].content.parts:
        if part.text:
            print(f"Text: {part.text}")
            # For Jupyter: display(Markdown(part.text))
        elif part.executable_code:
            print(f"Executable Code (generated by model):\n```python\n{part.executable_code.code}\n```")
            # For Jupyter: display(Markdown(f"```python\n{part.executable_code.code}\n```"))
        elif part.inline_data: # For generated images/plots from code
            print("Generated data (e.g., image from plot) found.")
            # For Jupyter: display(IPImage(data=part.inline_data.data))
            # To save:
            # with open("generated_plot.png", "wb") as f:
            #     f.write(part.inline_data.data)
            # print("Plot saved as generated_plot.png")
    # Final textual summary often in code_exec_response.text, if the model provides one after execution.
    if code_exec_response.text and not any(p.text for p in code_exec_response.candidates[0].content.parts if p.text): # If text isn't already in parts
        print(f"Overall Textual Response: {code_exec_response.text}")

except Exception as e:
    print(f"Code Execution tool error: {e}")
```
    </doc>
    <doc title="Model Context Protocol (MCP)" desc="Integrating with external tools and services via MCP, a standardized remote protocol. Requires asynchronous operations.">
# Model Context Protocol (MCP)

MCP enables AI models to connect to remote servers providing tools and data, offering a scalable and standardized way to extend capabilities. MCP interactions typically use the asynchronous `client.aio` interface.

```python
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from datetime import datetime

# Assume client and MODEL_ID_MCP (e.g., "gemini-2.0-flash" or "gemini-2.5-flash-preview-05-20") are defined.
# MODEL_ID_MCP = "gemini-2.0-flash" # As per workshop
# Requires an async environment (e.g., `await` in Jupyter or `asyncio.run()`).
```

## Stdio MCP Servers (Local Processes)
Communicate with MCP servers running as local processes via standard I/O.
```python
# Example: A hypothetical local weather MCP server
# Ensure the command 'npx -y @philschmid/weather-mcp' is runnable in the environment.
weather_server_params = StdioServerParameters(
    command="npx", # Or python, or other executable
    args=["-y", "@philschmid/weather-mcp"], # Example command from workshop
)

async def run_stdio_mcp():
    # This function needs to be run in an async context, e.g. await run_stdio_mcp() in Jupyter
    # or asyncio.run(run_stdio_mcp()) in a script.
    print("Attempting to run Stdio MCP example...")
    try:
        async with stdio_client(weather_server_params) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize() # Discover tools from server
                print(f"Stdio MCP tools: {mcp_session.tool_schemas}")

                mcp_prompt = f"What's the weather in Berlin on {datetime.now().strftime('%Y-%m-%d')}?"
                mcp_config = types.GenerateContentConfig(tools=[mcp_session]) # Pass session as a tool

                response = await client.aio.models.generate_content(
                    model=f"models/{MODEL_ID_MCP}", # Use an async-compatible model
                    contents=mcp_prompt,
                    config=mcp_config
                )
                print(f"Stdio MCP Response: {response.text}")
                # Inspect response.automatic_function_calling_history for tool interactions
                if response.automatic_function_calling_history:
                    print("Automatic Function Calling History (MCP):")
                    for event in response.automatic_function_calling_history:
                        if event.parts[0].function_call:
                            fc = event.parts[0].function_call
                            print(f"  - Call: {fc.name}({dict(fc.args)})")
                        elif event.parts[0].function_response:
                            fr = event.parts[0].function_response
                            print(f"  - Resp from {fr.name}: {fr.response}")
    except Exception as e:
        print(f"Stdio MCP error: {e}. Ensure the MCP server command is valid and runnable.")
# To run (example):
# if __name__ == '__main__':
#     asyncio.run(run_stdio_mcp())
```

## Remote HTTP MCP Servers
Connect to MCP servers hosted over HTTP.
```python
# Example: DeepWiki MCP server (from workshop)
deepwiki_mcp_url = "https://mcp.deepwiki.com/mcp"

async def run_remote_mcp_chat():
    # This function needs to be run in an async context.
    print(f"Attempting to run Remote MCP Chat example with {deepwiki_mcp_url}...")
    try:
        async with streamablehttp_client(deepwiki_mcp_url) as (read, write, _headers):
            async with ClientSession(read, write) as remote_session:
                await remote_session.initialize()
                print(f"Remote MCP tools (DeepWiki): {remote_session.tool_schemas}")

                remote_mcp_config = types.GenerateContentConfig(tools=[remote_session])
                mcp_chat = await client.aio.chats.create( # Use await for async chat creation
                    model=f"models/{MODEL_ID_MCP}",
                    config=remote_mcp_config
                )
                print("DeepWiki MCP Chat Agent: Ask about GitHub repos (e.g., 'Tell me about google/generative-ai-python'). Type 'exit' to quit.")
                while True:
                    try:
                        user_query = input("You: ") # input() is blocking, better for CLI examples
                    except EOFError: # Handle if input stream ends
                        break
                    if user_query.lower() == "exit": break
                    
                    chat_response = await mcp_chat.send_message(user_query)
                    print(f"Agent: {chat_response.text}")
                    
                    # Optionally print chat_response.automatic_function_calling_history
                    if chat_response.automatic_function_calling_history:
                        history_to_print = chat_response.automatic_function_calling_history
                        # Workshop logic to remove user input if it's the first item in history
                        if history_to_print and history_to_print[0].parts[0].text == user_query:
                            history_to_print.pop(0)
                        if history_to_print:
                            print("  (MCP Tool History for this turn):")
                            for call_event in history_to_print:
                                if call_event.parts[0].function_call:
                                    fc = call_event.parts[0].function_call
                                    print(f"    - Tool Call: {fc.name}({dict(fc.args)})")
                                elif call_event.parts[0].function_response:
                                    fr = call_event.parts[0].function_response
                                    response_summary = str(fr.response.get('result', 'N/A'))
                                    print(f"    - Tool Response from {fr.name}: {response_summary[:100]}{'...' if len(response_summary)>100 else ''}")
    except Exception as e:
        print(f"Remote MCP error: {e}")
# To run (example):
# if __name__ == '__main__':
#      asyncio.run(run_remote_mcp_chat())
```
For MCP, the `session` object obtained from `ClientSession` is passed in the `tools` list of `GenerateContentConfig`. The `client.aio` SDK then handles the asynchronous communication and tool invocation with the MCP server. The `response.automatic_function_calling_history` attribute can show the sequence of tool calls and responses.
    </doc>
</docs>

<api>
    <doc title="Core Client and Model Interaction" desc="Primary classes and methods for initializing the client and interacting with Gemini models for content generation and token counting.">
### `genai.Client(api_key, client_options, transport)`
Initializes the client to interact with the Gemini API.
**Parameters:**
- `api_key` (str): The Gemini API key.
- `client_options` (optional): Advanced client configuration options.
- `transport` (optional): gRPC transport specification (e.g., 'grpc', 'rest').

### `client.models.list()`
Lists available models accessible with the configured API key.
**Returns:** An iterable of `Model` objects.

### `client.models.get(name_or_id)`
Retrieves details for a specific model.
**Parameters:** `name_or_id` (str): Model name (e.g., "models/gemini-1.5-flash-latest" or "gemini-1.5-flash-latest").
**Returns:** A `Model` object.

### `client.models.generate_content(model, contents, generation_config, safety_settings, tools, tool_config)`
Generates content in a single, non-streaming call.
**Parameters:**
- `model` (str): Model ID (e.g., "models/gemini-1.5-flash-latest").
- `contents` (str | Iterable[Part | str | dict | File]): Prompt or multimodal content.
- `generation_config` (types.GenerateContentConfig | dict, optional): Generation parameters.
- `safety_settings` (Iterable[types.SafetySetting], optional): Content safety filters.
- `tools` (Iterable[types.Tool | types.FunctionDeclaration | Callable], optional): Tools for function calling or native capabilities.
- `tool_config` (types.ToolConfig, optional): Configuration for tool usage.
**Returns:** `types.GenerateContentResponse`. Access `.text`, `.parts`, `.parsed` (for structured output), `.usage_metadata`.

### `client.models.generate_content_stream(model, contents, ...)`
Generates content as a stream for incremental responses. Parameters are similar to `generate_content`.
**Returns:** An iterable of `types.GenerateContentResponse` chunks.

### `client.models.count_tokens(model, contents)`
Counts tokens for the given content and model.
**Parameters:** `model` (str), `contents` (str | Iterable[Part | str | dict | File]).
**Returns:** `types.CountTokensResponse` (with `total_tokens` attribute).

### `client.aio.models.generate_content(model, contents, ...)`
Asynchronous version of `generate_content`. Used with `await`. Necessary for MCP.
    </doc>
    <doc title="Chat Functionality" desc="Classes and methods for managing multi-turn conversations.">
### `client.chats.create(model, history, config, safety_settings)`
Creates a new chat session.
**Parameters:**
- `model` (str): Model ID for the chat.
- `history` (Iterable[types.Content], optional): Initial conversation history.
- `config` (types.GenerateContentConfig | dict, optional): Default config for the chat (e.g., `system_instruction`).
- `safety_settings` (Iterable[types.SafetySetting], optional): Default safety settings.
**Returns:** `types.ChatSession`.

### `chat_session.send_message(message, config, safety_settings, tools, tool_config)`
Sends a message in the chat and gets a response.
**Parameters:** `message` (str | Iterable[Part | str | dict | File]), other parameters override session defaults.
**Returns:** `types.GenerateContentResponse`.

### `chat_session.history` or `chat_session.get_history()`
Retrieves the conversation history as a list of `types.Content` objects.

### `client.aio.chats.create(...)` and `await chat_session.send_message(...)`
Asynchronous versions for use with `await`, particularly with MCP tools.
    </doc>
    <doc title="File API" desc="Methods for managing files (upload, get, delete, list) to be used as context with Gemini models.">
### `client.files.upload(path, display_name, mime_type)`
Uploads a file.
**Parameters:**
- `path` (str | pathlib.Path | io.BytesIO | io.BufferedReader): File path or file-like object.
- `display_name` (str, optional): User-friendly name.
- `mime_type` (str, optional): MIME type (SDK attempts to infer).
**Returns:** `types.File` object (attributes: `name`, `uri`, `mime_type`, `state`, etc.).

### `client.files.get(name)`
Retrieves metadata for an uploaded file, useful for checking processing `state`.
**Parameters:** `name` (str): The file resource name (e.g., `uploaded_file.name`).
**Returns:** `types.File`.

### `client.files.delete(name)`
Deletes an uploaded file.
**Parameters:** `name` (str): File resource name.

### `client.files.list(page_size, page_token)`
Lists uploaded files.
**Returns:** Iterable of `types.File` objects.
    </doc>
    <doc title="Key Data Structures (google.genai.types)" desc="Essential types for constructing prompts, configurations, and interpreting responses.">
### `types.Part`
A component of `Content`. Created via:
- `types.Part.from_text(str)`
- `types.Part.from_bytes(bytes, mime_type: str)` (e.g., for local images)
- `types.Part.from_uri(uri: str, mime_type: str)` (for File API files, YouTube videos)
- `types.Part.from_function_call(FunctionCall)` (model requests a call)
- `types.Part.from_function_response(name: str, response: dict)` (result of a call)
- Or `types.Part(inline_data=types.Blob(mime_type="...", data=...))`

### `types.Content`
Represents one turn/message. Attributes: `role` (str: "user", "model", "function"), `parts` (List[Part]).

### `types.GenerateContentConfig`
Configuration for generation. Parameters include `temperature`, `max_output_tokens`, `top_p`, `top_k`, `stop_sequences`, `system_instruction`, `response_mime_type`, `response_schema` (for Pydantic model), `response_modalities` (e.g., `['IMAGE']`, `['AUDIO']`), `speech_config`, `tools`, `tool_config`.

### `types.Tool`
Defines a tool.
- Custom functions: `types.Tool(function_declarations=[...])`
- Native Search: `types.Tool(google_search=types.GoogleSearch())`
- Native URL Context: `types.Tool(url_context=types.UrlContext())`
- Native Code Execution: `types.Tool(code_execution={})`
- MCP Session: The `mcp.ClientSession` object itself is passed in the `tools` list.

### `types.FunctionDeclaration`
Schema for a custom function. Attributes: `name` (str), `description` (str), `parameters` (types.Schema).

### `types.Schema`
Defines data structure (for function parameters, structured output). Attributes: `type` (types.Type enum), `properties` (dict), `items` (Schema), `enum` (list), `description`, `required` (list).

### `types.FileData`
Used with `types.Part` for file URIs, e.g., `types.Part(file_data=types.FileData(file_uri=youtube_url))`.

### `types.SpeechConfig`, `types.VoiceConfig`, `types.PrebuiltVoiceConfig`
Configuration for Text-to-Speech, specifying voice characteristics. `PrebuiltVoiceConfig(voice_name="...")`.

### `Pydantic Models`
Standard Pydantic models are used with `response_schema` in `GenerateContentConfig` for defining structured JSON output schemas.
    </doc>
    <doc title="Response Objects" desc="Structure of API responses.">
### `types.GenerateContentResponse`
Object returned by generation methods.
**Key Attributes:**
- `text` (str): Concatenated text from the first candidate's parts.
- `parts` (List[Part]): Parts from the first candidate.
- `candidates` (List[Candidate]): List of generated candidates. Each `Candidate` has `.content` (a `types.Content` object), `.finish_reason`, `.safety_ratings`, `.citation_metadata`.
- `prompt_feedback` (PromptFeedback): Feedback on the input prompt (e.g., block reason).
- `usage_metadata` (UsageMetadata): Token counts (`prompt_token_count`, `candidates_token_count`, `total_token_count`, `thoughts_token_count`).
- `parsed` (Any): The Pydantic object if structured output was successful.
- `automatic_function_calling_history` (List[Content]): History of automatic tool/MCP calls.
    </doc>
    <doc title="Model Context Protocol (MCP) Specifics" desc="Key components for MCP integration from `mcp` library and `google.genai`.">
### `mcp.ClientSession(read_stream, write_stream)`
Manages MCP communication. Use `async with ClientSession(...) as session:`.

### `await session.initialize()`
Initializes the MCP session, discovering server tools (`session.tool_schemas`).

### `mcp.client.stdio.StdioServerParameters(command, args, env)`
Config for local Stdio MCP servers.

### `mcp.client.stdio.stdio_client(params)`
Context manager for Stdio MCP server connection. Returns `(read_stream, write_stream)`.

### `mcp.client.streamable_http.streamablehttp_client(url, headers)`
Context manager for remote HTTP MCP server connection. Returns `(read_stream, write_stream, response_headers)`.

**MCP Tool Usage with Gemini SDK:**
The initialized `mcp.ClientSession` object is passed directly in the `tools` list of `types.GenerateContentConfig` when making calls with `client.aio.models.generate_content` or `client.aio.chats.create / await chat.send_message`.
    </doc>
</api>

<examples>
    <!-- Note: Examples below assume 'client' and relevant 'MODEL_ID's are initialized as described in earlier sections. -->
    <!-- Full runnable code might require additional setup like specific file paths or error handling. -->
    <doc title="Chat with a Book (Text &amp; File API)" desc="Demonstrates uploading a text file (book) and conversing about its content using a chat session with system instructions.">
```python
from google import genai
from google.genai import types
import os
import requests # For downloading the book

# Assumes client and MODEL_ID ("gemini-2.5-flash-preview-05-20" or similar) are pre-configured globally.
# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# MODEL_ID = "gemini-2.5-flash-preview-05-20"

book_file_path = "alice_in_wonderland.txt"
if not os.path.exists(book_file_path):
    book_text_url = "https://www.gutenberg.org/files/11/11-0.txt"
    try:
        response_book_req = requests.get(book_text_url)
        response_book_req.raise_for_status()
        with open(book_file_path, "w", encoding="utf-8") as f:
            f.write(response_book_req.text)
        print(f"Book '{book_file_path}' downloaded successfully!")
    except requests.RequestException as e:
        print(f"Error downloading book: {e}")
        # Handle error, e.g. exit if book is essential for the example.
        # For this illustrative purpose, we'll proceed, but a real app would need the file.

if os.path.exists(book_file_path):
    try:
        print(f"Uploading {book_file_path}...")
        uploaded_book_file = client.files.upload(path=book_file_path, display_name="Alice in Wonderland")
        print(f"Book uploaded: {uploaded_book_file.name}")

        chat_config = types.GenerateContentConfig(
            system_instruction="You are an expert book reviewer with a witty and engaging tone. Provide insightful analysis while keeping responses accessible and entertaining.",
            temperature=0.8,
        )
        book_chat = client.chats.create(
            model=f"models/{MODEL_ID}",
            config=chat_config
        )

        initial_prompt_text = f"Summarize the book in 5 key bullet points, then ask me what I'd like to discuss further."
        print(f"\nUser: (Asking for summary of Alice in Wonderland)")

        response_initial = book_chat.send_message(
            message=[initial_prompt_text, uploaded_book_file] # Pass text and file object
        )
        print(f"Book Reviewer Bot:\n{response_initial.text}")

        followup_question = "Tell me more about the Mad Hatter's tea party. What was so peculiar about it?"
        print(f"\nUser: {followup_question}")

        response_followup = book_chat.send_message(message=followup_question)
        print(f"Book Reviewer Bot:\n{response_followup.text}")

        # Optional: client.files.delete(name=uploaded_book_file.name)
        # print(f"\nDeleted file: {uploaded_book_file.name}")

    except Exception as e:
        print(f"Error in Chat with Book example: {e}")
else:
    print(f"Skipping Chat with Book example as '{book_file_path}' not found.")
```
    </doc>
    <doc title="Product Description from Image (Multimodal)" desc="Uses an image (from URL bytes) as input to generate a product description, features, and slogan.">
```python
from google.genai import types
import requests
from io import BytesIO
# from PIL import Image # To display image if in notebook
# Assumes client and MODEL_ID ("gemini-2.5-flash-preview-05-20" or similar multimodal) are pre-configured.

product_image_url = "https://images.unsplash.com/file-1705123271268-c3eaf6a79b21image?w=416&dpr=2&auto=format&fit=crop&q=60" # Example chair

try:
    print(f"Fetching image from: {product_image_url}")
    img_response = requests.get(product_image_url)
    img_response.raise_for_status()
    image_data_bytes = img_response.content

    product_image_part = types.Part.from_bytes(data=image_data_bytes, mime_type="image/jpeg") # Assuming JPEG

    prompt_text_product = """
Based on the image provided:
1. Identify the main product shown.
2. Describe its key visual features (color, material if discernible, style).
3. Suggest 2-3 potential use cases for this product.
4. Write a short, catchy marketing slogan for it.
"""
    prompt_part_product = types.Part.from_text(prompt_text_product)

    print("Asking Gemini for product description...")
    response_product_desc = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=[prompt_part_product, product_image_part] # Text prompt + Image part
    )

    print("\nGenerated Product Description:")
    print(response_product_desc.text)

except requests.RequestException as e:
    print(f"Error fetching image: {e}")
except Exception as e:
    print(f"Error generating product description: {e}")
```
    </doc>
    <doc title="Avatar Generation (Image Gen &amp; TTS)" desc="Combining image generation to create an avatar and text-to-speech to give it its voice, using specific models for each task.">
```python
from google.genai import types
from PIL import Image
from io import BytesIO
import numpy as np
import soundfile as sf
# Assumes client is pre-configured.
IMAGE_GEN_MODEL_ID = "gemini-2.0-flash-preview-image-generation"
TTS_MODEL_ID = "gemini-2.5-flash-preview-tts"

# 1. Generate an Avatar Image
prompt_avatar_image = "Generate an image of a friendly, futuristic robot assistant with a welcoming smile, digital art style, high resolution."
generated_avatar_image_obj = None
avatar_image_saved = False

print(f"Requesting avatar image with prompt: '{prompt_avatar_image}'")
try:
    response_img_gen = client.models.generate_content(
        model=f"models/{IMAGE_GEN_MODEL_ID}",
        contents=prompt_avatar_image,
        config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    )
    for part in response_img_gen.candidates[0].content.parts:
        if part.text: print(f"Image Gen Text: {part.text}")
        elif part.inline_data and part.inline_data.mime_type.startswith('image/'):
            generated_avatar_image_obj = Image.open(BytesIO(part.inline_data.data))
            generated_avatar_image_obj.save("generated_avatar.png")
            avatar_image_saved = True
            print("Avatar image saved as generated_avatar.png")
            break # Assuming one image is generated
    if not avatar_image_saved: print("Failed to generate or save avatar image from response.")
except Exception as e:
    print(f"Error generating image: {e}")

# 2. Create an Introduction Text
avatar_introduction_text = "Hello! I am Vision, your friendly AI assistant. I'm excited to help you generate amazing things!"

# 3. Generate Speech for the Introduction
if avatar_image_saved: # Only proceed if image was successfully processed
    tts_prompt_content = f"Say in a voice suitable for this character described as '{prompt_avatar_image}': {avatar_introduction_text}"
    print(f"\nRequesting speech for: '{avatar_introduction_text}'")
    try:
        response_speech = client.models.generate_content(
           model=f"models/{TTS_MODEL_ID}",
           contents=tts_prompt_content,
           config=types.GenerateContentConfig(
              response_modalities=["AUDIO"],
              speech_config=types.SpeechConfig(
                 voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
                 )
              )
           )
        )
        if response_speech.candidates[0].content.parts[0].inline_data:
            audio_data_bytes = response_speech.candidates[0].content.parts[0].inline_data.data
            audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
            sf.write("avatar_introduction.wav", audio_array, 24000) # 24kHz sample rate
            print("Avatar introduction speech saved as avatar_introduction.wav")
        else: print("Failed to generate speech audio from response.")
    except Exception as e:
        print(f"Error generating speech: {e}")
else:
    print("Skipping speech generation as avatar image was not generated/saved.")
```
    </doc>
    <doc title="PDF to Structured Data (Structured Output &amp; File API)" desc="Extracts structured information (like invoice details) from a PDF document into a Pydantic model.">
```python
from google.genai import types
from pydantic import BaseModel
from typing import List
import os
# Assumes client and MODEL_ID ("gemini-2.5-flash-preview-05-20" or similar) are pre-configured.

class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    vendor_address: str
    total_amount: float
    items: List[InvoiceItem]

# Ensure '../assets/data/rewe_invoice.pdf' (from workshop) or a similar PDF exists.
pdf_file_path_struct = "../assets/data/rewe_invoice.pdf" # Path from workshop
if not os.path.exists(pdf_file_path_struct):
    print(f"PDF file {pdf_file_path_struct} not found. Cannot run PDF to Structured Data example.")
else:
    try:
        print(f"Uploading PDF: {pdf_file_path_struct}")
        uploaded_pdf = client.files.upload(path=pdf_file_path_struct, display_name="REWE Invoice")
        print(f"PDF uploaded: {uploaded_pdf.name}")

        extraction_prompt = "Extract all invoice information from this PDF including items, vendor details, and totals. Ensure the output matches the provided schema."

        print("Requesting structured data extraction from PDF...")
        response_structured_pdf = client.models.generate_content(
            model=f"models/{MODEL_ID}",
            contents=[extraction_prompt, uploaded_pdf], # Prompt text + uploaded PDF file object
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceData, # Pass the Pydantic model class
            ),
        )

        if hasattr(response_structured_pdf, 'parsed') and response_structured_pdf.parsed:
            invoice_data: InvoiceData = response_structured_pdf.parsed
            print("\nExtracted Invoice Data (Structured):")
            print(f"Invoice #: {invoice_data.invoice_number}")
            print(f"Vendor: {invoice_data.vendor_name} at {invoice_data.vendor_address}")
            print(f"Date: {invoice_data.date}")
            print(f"Total Amount: {invoice_data.total_amount}")
            print("Items:")
            for item in invoice_data.items:
                print(f"  - {item.description}: {item.quantity} x {item.unit_price} = {item.total}")
        else:
            print("Failed to get parsed structured data from PDF. Raw response (if any):")
            print(response_structured_pdf.text)
        # client.files.delete(name=uploaded_pdf.name) # Optional cleanup
    except Exception as e:
        print(f"An error occurred in PDF to Structured Data: {e}")
```
    </doc>
    <doc title="Calculator Agent (Automatic Function Calling)" desc="Building a simple calculator using Gemini's automatic function calling feature. The SDK handles the interaction with Python functions defined with type hints and docstrings.">
```python
from google.genai import types
# Assumes client and MODEL_ID ("gemini-2.5-flash-preview-05-20" or similar) are pre-configured.

def add(a: float, b: float) -> dict:
    """Add two numbers. Args: a (float): First number. b (float): Second number. Returns: Sum of the numbers."""
    result = a + b
    print(f"CALLED: add({a}, {b}) -> {result}")
    return {"operation": "addition", "result": result}

def subtract(a: float, b: float) -> dict:
    """Subtract second number from first. Args: a (float): First number. b (float): Second number. Returns: Difference."""
    result = a - b
    print(f"CALLED: subtract({a}, {b}) -> {result}")
    return {"operation": "subtraction", "result": result}

def multiply(a: float, b: float) -> dict:
    """Multiply two numbers. Args: a (float): First number. b (float): Second number. Returns: Product."""
    result = a * b
    print(f"CALLED: multiply({a}, {b}) -> {result}")
    return {"operation": "multiplication", "result": result}

def divide(a: float, b: float) -> dict:
    """Divide first number by second. Args: a (float): Dividend. b (float): Divisor. Returns: Quotient or error if division by zero."""
    if b == 0:
        print(f"CALLED: divide({a}, {b}) -> Error: Division by zero")
        return {"operation": "division", "error": "Division by zero"}
    result = a / b
    print(f"CALLED: divide({a}, {b}) -> {result}")
    return {"operation": "division", "result": result}

calculator_tools = [add, subtract, multiply, divide] # List of callable Python functions
calculator_config = types.GenerateContentConfig(tools=calculator_tools)

prompt_single_op = "What is 15 multiplied by 7?"
print(f"\nUser: {prompt_single_op}")
try:
    response_single = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=prompt_single_op,
        config=calculator_config
    )
    print(f"Calculator Agent: {response_single.text}")
except Exception as e:
    print(f"Error during single op calculation: {e}")

prompt_complex_op = "Calculate (25 + 15) * 3 - 10. Please show the steps if you can."
print(f"\nUser: {prompt_complex_op}")
try:
    response_complex = client.models.generate_content(
        model=f"models/{MODEL_ID}",
        contents=prompt_complex_op,
        config=calculator_config # SDK handles multiple function calls if needed
    )
    print(f"Calculator Agent: {response_complex.text}")
except Exception as e:
    print(f"Error during complex calculation: {e}")
```
    </doc>
</examples>
</project>