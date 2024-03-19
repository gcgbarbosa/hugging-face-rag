from src.chatbot import ChatBot

cb = ChatBot("TonicBot")

# question = 'código atc furosemida?'

question = ""
while question != "quit":
    print("Type your question: ", end="")
    question = input()
    response = cb.respond(question)

    print("#####################")
    print("\n\n### Question:\n" + question, end="\n\n")
    print("### Response:\n" + response)
    print("#####################", end="\n\n")
