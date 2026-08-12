"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit

Architecture: IPO Model (Input -> Process -> Output)
- Input Loop: continuous while loop
- Sanitization: lowercase + strip whitespace
- Knowledge Base: dictionary (O(1) lookup instead of if-elif ladder)
- Fallback: default response via .get()
- Exit Strategy: dedicated exit command breaks the loop
"""

# ---- Knowledge Base ----
# Each key is a normalized intent, each value is the bot's response.
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bunch of if-else logic, but I'm doing great!",
    "what is your name": "I'm ChatBot, your friendly rule-based assistant.",
    "who made you": "I was built as Project 1 for the DecodeLabs AI Internship.",
    "what can you do": "I can respond to a few predefined commands. Try 'help' to see them.",
    "help": "Try: hello, how are you, what is your name, who made you, what can you do, bye.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

# Commands that end the conversation
exit_commands = {"bye", "exit", "quit", "goodbye"}

FALLBACK_RESPONSE = "I do not understand. Type 'help' to see what I can do."
FAREWELL_RESPONSE = "Goodbye! Have a great day."


def sanitize(raw_input: str) -> str:
    """Normalize user input: lowercase and strip surrounding whitespace."""
    return raw_input.lower().strip()


def get_response(clean_input: str) -> str:
    """Look up the response for a given intent, with a fallback for unknowns."""
    return responses.get(clean_input, FALLBACK_RESPONSE)


def run_chatbot():
    print("ChatBot: Hello! I'm your rule-based assistant. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        if clean_input in exit_commands:
            print(f"ChatBot: {FAREWELL_RESPONSE}")
            break

        reply = get_response(clean_input)
        print(f"ChatBot: {reply}")


if __name__ == "__main__":
    run_chatbot()
