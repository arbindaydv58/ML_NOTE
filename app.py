from src.chatbot import ask

print("""
=========================================
 ML NOTE CHATBOT (CHATGPT STYLE)
 type 'exit' to quit
 type 'debug on' to see retrieved chunks
=========================================
""")

debug=False

while True:

    q=input("You: ")

    if q.lower()=="exit":
        break

    if q.lower()=="debug on":
        debug=True
        print("Debug enabled")
        continue

    if q.lower()=="debug off":
        debug=False
        print("Debug disabled")
        continue

    print("\nAssistant:",ask(q,debug),"\n")