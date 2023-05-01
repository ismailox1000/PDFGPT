import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os 
from streamlit_chat import message 
from utils import get_initial_message, update_chat

# Define Streamlit app
def main():
    st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖")
    st.title("PDF Question-Answering Chatbot 🤖 ")
    st.markdown("<br>", unsafe_allow_html=True)
    

    # API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    os.environ["OPENAI_API_KEY"]=api_key
    st.markdown("<br>", unsafe_allow_html=True)
    # Check if API key is valid
    if api_key and len(api_key)==51 :  

        st.subheader("Get Answers from Your PDF with OpenAI")
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
        
        if uploaded_file is not None:
            embeddings = OpenAIEmbeddings()
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chain = load_qa_chain(OpenAI(api_key=api_key), chain_type="stuff")
            texts = mamatext(uploaded_file, text_splitter)
            docsearch = FAISS.from_texts(texts, embeddings)

            if "generated" not in st.session_state:
                st.session_state["generated"] = []
            if "past" not in st.session_state:
                st.session_state["past"] = []

            query = st.text_input("Type Your Question : ", key="input")

            if "messages" not in st.session_state:
                st.session_state["messages"] = get_initial_message()

            if query:
                docs = docsearch.similarity_search(query)
                chain.run(input_documents=docs, question=query)
                with st.spinner("generating..."):
                    messages = st.session_state["messages"]
                    messages = update_chat(messages, "user", query)
                    response = chain.run(input_documents=docs, question=query)
                    messages = update_chat(messages, "assistant", response)
                    st.session_state.past.append(query)
                    st.session_state.generated.append(response)

            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                    message(st.session_state["generated"][i], key=str(i))

                with st.expander("Show Messages"):
                    st.write(messages)

        else:
            st.warning("Please upload a PDF file to continue.")

    else:
        st.warning("Please enter your OpenAI API key to access the content.")


def mamatext(uploaded_file, text_splitter):
    pdf_contents = uploaded_file.read() # read file content here
    reader = PdfReader(io.BytesIO(pdf_contents))
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    texts = text_splitter.split_text(raw_text)
    
    return texts

if __name__ == "__main__":
    main()
