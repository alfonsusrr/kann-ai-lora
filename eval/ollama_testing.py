import ollama

with open('./basemodel/ModelfileNene', 'r') as file:
    model = file.read()
    
ollama.create(model="NeneEval", modelfile=model)

response = ollama.chat(model="NeneEval", messages=[
    {
        'role': 'user',
        'content': 'Hello, how are you?'
    }
])

print(response['message']['content'])