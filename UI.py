from main import get_generated_text
# from Data_Sources_Utils.vectorDB_router import load_vectordb
import streamlit as st
import random, time
def initialize_counter():
    # This could be a more complex initialization if needed
    print("Counter initialized")
    return 0
# Streamed response emulator
def get_LLM_response(retrive_data, context, query):
    # print("---------------------------")
    # print("Prompt:", txt)
    # print("---------------------------")
    # prompt = context + "<|start_header_id|>user<|end_header_id|>" + query + "<|eot_id|>"
    return retrive_data(context, query)
    # return get_generated_text("meta-llama/llama-3-8b-instruct", prompt)
def stream_response(retrive_data, context, query):
    resp = get_LLM_response(retrive_data, context, query)
    for word in resp.split(" "):
        yield word + " "
        time.sleep(0.01)
def run_chatbot(get_generated_text):
    st.title("Branched RAG")
    # if 'counter' not in st.session_state:
    #     st.session_state.counter = initialize_counter()
    # st.write("Chat Length:", st.session_state.counter)
    # Increment the counter
    # st.session_state.counter += 1
    # if 'vectordb' not in st.session_state:
    #     if country_list_flag:
    #         st.session_state.vectordb, st.session_state.country_list = load_vectordb(True)
    #     else:
    #         st.session_state.vectordb = load_vectordb()
    # print("Logging: vectordb loaded")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Generate the text stream with tags
        text_stream = ""
        for message in st.session_state.messages:
            if message['role'] == 'user':
                text_stream += "<|start_header_id|>user<|end_header_id|>\n\n" + message['content'] + "<|eot_id|>"
            elif message['role'] == 'assistant':
                text_stream += "<|start_header_id|>assistant<|end_header_id|>\n\n" + message['content'] + "<|eot_id|>"
        # print("=====================================")
        # print("text_stream:", text_stream)
        # print("=====================================")
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(stream_response(get_generated_text, text_stream, prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # print("st.session_state.messages:", st.session_state.messages)
run_chatbot(get_generated_text)