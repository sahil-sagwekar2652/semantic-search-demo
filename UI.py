import streamlit as st
import PyPDF2



def extract_text_from_pdf(_pdf):
    pdf_reader = PyPDF2.PdfReader(_pdf)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text
def UI():
    st.title("Title")    #add title here of the site

    PDF = st.file_uploader("Choose a PDF file", type="pdf")

    if PDF is not None:     #to check the PDF file type
        st.success("File successfully uploaded!")
        
        

        #pdf_text contains the text content of the uploaded pdf file in the form of normal python string 
        pdf_text = extract_text_from_pdf(PDF)#so this can be to passed to ML models after importing them in this file
        st.write("PDF Content:")
        st.write(pdf_text)
        
        
        st.markdown("<h1>Write Your Question Prompt here:</h1>" , unsafe_allow_html=True)
        user_input = st.text_area("Type your text here:",value="" )
        if st.button("Submit"):
            #write the ML Code in this Block
            st.write("You entered:", user_input)#use this Function to write the output


UI()