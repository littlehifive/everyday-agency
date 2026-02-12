import os
from bs4 import BeautifulSoup
import pandas as pd
import sys

# Increase recursion depth just in case, though iterative is better
sys.setrecursionlimit(10000)

INPUT_FILE = 'data/chat.html'
OUTPUT_FILE = 'data/chat_messages.parquet'

def parse_chat_html(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print("Reading file... (this may take a while)")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Extracting JSON data...")
    # diverse ways to define the variable start
    start_marker = 'var jsonData = '
    start_index = content.find(start_marker)
    
    if start_index == -1:
        print("Error: Could not find 'var jsonData =' in file.")
        return

    # The JSON data starts after the marker
    json_start = start_index + len(start_marker)
    
    # We can just take the rest of the file from this point.
    # json.JSONDecoder().raw_decode() will parse one valid JSON object and ignore the rest.
    
    json_str_full = content[json_start:]
    
    import json
    try:
        print("Parsing JSON string using raw_decode...")
        decoder = json.JSONDecoder()
        data, end_idx = decoder.raw_decode(json_str_full)
        print(f"Values parsed successfully. JSON end at index {end_idx}.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Try to print a snippet around the error
        return

    print(f"Found {len(data)} conversations.")

    all_messages = []

    for i, conv in enumerate(data):
        title = conv.get('title', 'Unknown Conversation')
        conversation_id = conv.get('id', f"conv_{i}")
        mapping = conv.get('mapping', {})
        
        # Traverse the mapping. It's a dict of nodes.
        # We can just iterate over all values in mapping as they contain the messages.
        # If order matters, we'd follow parent/children links, but for bulk export,
        # just dumping all messages is usually sufficient.
        # If we really want time order, we can sort by create_time later.
        
        for node_id, node in mapping.items():
            message = node.get('message')
            if not message:
                continue
                
            # Extract checks
            if message.get('author', {}).get('role') == 'system':
                continue # Skip system messages if desired? User usually wants their chat.
                # Actually, let's keep everything but maybe mark it.
            
            author_role = message.get('author', {}).get('role')
            create_time = message.get('create_time')
            
            content_parts = message.get('content', {}).get('parts', [])
            text_content = ""
            if content_parts:
                text_content = "\n".join([str(p) for p in content_parts if isinstance(p, str)])
            
            all_messages.append({
                'conversation_id': conversation_id,
                'conversation_title': title,
                'message_id': message.get('id'),
                'author_role': author_role,
                'create_time': create_time,
                'text': text_content
            })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} conversations...", end='\r')

    print(f"\nTotal messages extracted: {len(all_messages)}")

    if not all_messages:
        print("No messages found.")
        return

    df = pd.DataFrame(all_messages)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
    print("Done.")

if __name__ == "__main__":
    parse_chat_html(INPUT_FILE)
