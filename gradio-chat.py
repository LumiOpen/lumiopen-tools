import argparse
import gradio as gr
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def gen_ua_prompt(history, tokenizer):
    messages = [{"role": entry[0], "content": entry[1]} for entry in history]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # print("formatted prompt:", prompt)
    return prompt 


def respond(prompt, model, tokenizer):
    output = gen_answer(prompt, model, tokenizer)
    return output

def gen_answer(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")

    bad_words_ids = tokenizer.encode(['<NAME>', ' <NAME>'])

    try:
        with torch.no_grad():
            output = model.generate(
                    input_ids,
                    max_new_tokens=500,
                    #temperature=0.7,
                    #top_p=0.8,
                    #repetition_penalty=1.1,
                    #bad_words_ids=[[word] for word in bad_words_ids],
                    eos_token_id=tokenizer.eos_token_id,
                    #do_sample=True
                    )
        
        if output[0][-1] == tokenizer.eos_token_id:
            print("trimming final token")
            output = output[0][:-1]
        else:
            output = output[0]
        decoded_output = tokenizer.decode(output, skip_special_tokens=False)
        decoded_output = decoded_output[len(prompt):].lstrip()
    except torch.cuda.OutOfMemoryError as e:
        decoded_output = f"Something went wrong.  Please press Clear to continue"
    return decoded_output

def load(args):
    print("Cuda available:", torch.cuda.is_available())
    print("Loading tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
    )

    print("Loading model:", args.model)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",

    )
    diff = time.time() - start
    print(f"Loading model took", diff, "seconds")

    return model, tokenizer
    

def gradio_main(args):
    model, tokenizer = load(args)

    def process(message, history):
        print(history)
        print(message)

        entries = []

        # take last 5 message pairs = 10 messages
        for user, assistant in history[-5:]:
            entries.append(("user", user))
            entries.append(("assistant", assistant))
        entries.append(("user", message))

        prompt = gen_ua_prompt(entries, tokenizer)
        print(prompt)
        response = respond(prompt, model, tokenizer)
        print(response)
        return response

    gr.ChatInterface(
        process, 
        chatbot=gr.Chatbot(
            height=600,
            show_copy_button=True
        )).launch(share=True)


def main():
    model, tokenizer = load()

    # chat loop
    chat_history_length = 10
    history = []
    print(f"Welcome to LLM chat.  Using the last {chat_history_length} lines of history.\nType 'reset' to reset chat history.")
    while True:
        user_input = input("You: ").rstrip()
    
        if user_input == "reset":
            history = []
            print("")
            continue
        if user_input == "":
            continue
    
    
        history.append(("user", user_input))
        prompt = gen_ua_prompt(history, tokenizer)
        response = respond(prompt, model, tokenizer)
        print(f"Assistant:{response}")
        history.append(("assistant", response))
        history = history[-chat_history_length:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    if not args.text:
        print("Launching gradio...")
        gradio_main(args)
    else:
        print("Launching text chat...")
        main()
