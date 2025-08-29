from model import MusicChatbot

if __name__ == "__main__":
    bot = MusicChatbot("meta-llama/Llama-2-7b-chat-hf")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = bot.generate_response(user_input)
        print("Bot:", response)
