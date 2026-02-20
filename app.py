from src.chatbot import ask

print("\nML NOTE CHATBOT READY")
print("Type 'exit' to stop\n")

while True:

    q = input("You: ")

    if q.lower()=="exit":
        break

    print("\nBot:", ask(q),"\n")