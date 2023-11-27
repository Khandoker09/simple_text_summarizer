'''
Author: Khandoker Tanjim Ahammad
Date: 26.08.2023
Purpose: Simple text summarizer app using deep learning. 
generate requirement file: pipreqs --encoding=utf8 ./
'''
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Streamlit app layout
def smt5():
    st.title("Text Summarization App")
    st.sidebar.markdown("Maintained by: [Khandoker Tanjim Ahammad](https://github.com/Khandoker09)")
    st.markdown("###### This text summarizer was build using Hugging Face Transformer model. It is a open source library. From this library we have use T5-small model.The T5 (Text-to-Text Transfer Transformer) model is primarily an abstractive model. Abstractive models generate summaries or paraphrases in a more human-like manner by understanding the input text and generating new, coherent text that captures the essential meaning of the input. They have the ability to produce summaries that may not appear verbatim in the input text and can provide more concise and coherent summaries compared to extractive methods.")
    # Text input box
    st.markdown("###### Tips: Try to avoid the author name,image or table, only paste the the text you want to summarize. Also avoid adding text containing bullet points or list")
    
    st.subheader("Paste the text you want to summarize:")
    input_text = st.text_area("Input Text", "")

    if st.button("Summarize"):
        if input_text:
            # Tokenize input text and generate summary
            input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    smt5()
