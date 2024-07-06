import nltk
from nltk.chat.util import Chat, reflections

# Definisikan aturan percakapan untuk chatbot
rules = [
    (r'hi|hello|hey there', ['Hello!', 'Hey!', 'Hi there!']),
    (r'how are you?', ['I am good, thank you.', 'I\'m doing fine.', 'Pretty well, thanks!']),
    (r'what is your name?', ['I am a chatbot.', 'You can call me Chatbot.', 'My name is Chatbot.']),
    (r'(.*) your name(.*)', ['My name is Chatbot.', 'I\'m Chatbot.', 'You can call me Chatbot.']),
    (r'(.*) help (.*)', ['I can help you with various things.', 'Sure, I\'m here to help.', 'What do you need help with?']),
    (r'(.*) (location|city) ?', ['Tokyo, Japan', 'New York, USA', 'Berlin, Germany']),
    (r'bye|goodbye', ['Goodbye!', 'Bye!', 'Take care!']),
    (r'thanks|thank you', ['You\'re welcome!', 'No problem.', 'My pleasure!'])
]

# Fungsi untuk menjalankan chatbot
def chatbot():
    print("Hello! I'm Chatbot. How can I help you today?")

    chat = Chat(rules, reflections)
    while True:
        user_input = input("You: ")
        response = chat.respond(user_input)
        print("Chatbot:", response)
        
        # Keluar dari loop jika pengguna mengatakan "bye" atau "goodbye"
        if user_input.lower() in ['bye', 'goodbye']:
            break

if __name__ == "__main__":
    chatbot()
