import tkinter as tk
from tkinter import filedialog, scrolledtext
import openai
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())  # read local .env file
openai.api_key = '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'


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def collect_messages():
    prompt = input_text.get()
    input_text.delete(0, tk.END)
    context.append({'role': 'user', 'content': f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role': 'assistant', 'content': f"{response}"})
    conversation_area.insert(tk.END, f"User: {prompt}\n")
    conversation_area.insert(tk.END, f"Robot: {response}\n")
    conversation_area.insert(tk.END, "---------------------------\n")
    conversation_area.see(tk.END)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            context.append({'role': 'system', 'content': file_content})
        conversation_area.insert(tk.END, f"Loaded file: {file_path}\n")
        conversation_area.insert(tk.END, "---------------------------\n")
        conversation_area.see(tk.END)

def save_conversation():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[('Text Files', '*.txt')])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as file:
            conversation = conversation_area.get("1.0", tk.END)
            file.write(conversation)
        conversation_area.insert(tk.END, f"Saved conversation: {file_path}\n")
        conversation_area.insert(tk.END, "---------------------------\n")
        conversation_area.see(tk.END)

# Create the main window
window = tk.Tk()
window.title("Hakshe Conversation Testing")
window.geometry("600x500")

# Initialize the list to collect context messages
context = []

# Create the conversation area
conversation_area = scrolledtext.ScrolledText(window, height=20, width=60)
conversation_area.pack(pady=10)

# Create the input text widget
input_text = tk.Entry(window, width=50)
input_text.pack(pady=10)

# Create the send button
send_button = tk.Button(window, text="Send", command=collect_messages)
send_button.pack(side=tk.LEFT, padx=10)

# Create the load file button
load_file_button = tk.Button(window, text="Load File", command=load_file)
load_file_button.pack(side=tk.LEFT)

# Create the save conversation button
save_button = tk.Button(window, text="Save Conversation", command=save_conversation)
save_button.pack(side=tk.LEFT)

window.mainloop()
