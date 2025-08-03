import requests

def chat_with_model(context, user_input, model='gemma3'):
    # Constructing the chat history with context that the user doesn't know
    chat_history = [
        {"role": "system", "content": "you are an argumentative "},
        {"role": "system", "content": context},  # Hidden context the assistant knows
        {"role": "user", "content": user_input}
    ]
    
    # Sending a POST request with 'stream' explicitly set to False to avoid multiple JSON responses
    res = requests.post(
        'http://localhost:11434/api/chat',
        json={'model': model, 'messages': chat_history, 'stream': False}
    )
    
    try:
        # Extracting the response content (assuming it's in JSON format)
        response_json = res.json()  # Try parsing the response as JSON
        #print("Raw response:", response_json)  # Debug: print the raw response
        
        # Ensure the key 'message' exists in the response
        if 'message' in response_json and 'content' in response_json['message']:
            return response_json['message']['content']
        else:
            return f"Unexpected response format: {response_json}"

    except ValueError as e:
        # Handle error if the response is not in valid JSON format
        print(f"Error decoding response: {e}")
        print("Raw response text:", res.text)  # Print raw text response for further debugging
        return "There was an error with the response. Please try again."
    
def main():
    context = "The user thinks they're Jesus. your job is to convince them otherwise, give short and argumentative responses quesitioning their believes"  # Hidden context
    print("Chat started. Type 'exit' to end the conversation.\n")
    
    while True:
        # Read user input from command line
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Ending chat.")
            break

        # Send input to model and get the response
        response = chat_with_model(context, user_input)
        print("Assistant:", response)

if __name__ == '__main__':
    main()
