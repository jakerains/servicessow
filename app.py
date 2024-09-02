import streamlit as st
import os
import json
from openai import OpenAI
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO
import docx2txt
import PyPDF2
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import markdown
import base64
import mimetypes
from bs4 import BeautifulSoup
import whisper
from docx import Document
import PyPDF2

# Define the version number
VERSION = "1.4.0"  # This matches the latest version in the CHANGELOG.md

# Set page config at the very top, after imports
st.set_page_config(page_title="Sterling Services: S.O.W. Generator", page_icon="📄")

def detect_file_type(file):
    # Get the file extension
    file_extension = os.path.splitext(file.name)[1].lower()
    
    # Define file type based on extension
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac']
    text_extensions = ['.txt', '.rtf', '.doc', '.docx', '.pdf', '.odt', '.md']
    
    if file_extension in audio_extensions:
        return "Audio"
    elif file_extension in text_extensions:
        return "Text"
    else:
        # If the extension is not recognized, try to guess the MIME type
        mime_type, _ = mimetypes.guess_type(file.name)
        if mime_type:
            if mime_type.startswith('audio'):
                return "Audio"
            elif mime_type.startswith('text') or mime_type == 'application/pdf':
                return "Text"
    
    # If we can't determine the type, return None
    return None

def main():
    st.title("Sterling Services: S.O.W. Generator")
    
    # Configuration section
    st.subheader("Configuration")
    
    # Initialize session state
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''
    if 'api_key_set' not in st.session_state:
        st.session_state['api_key_set'] = False
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = 'gpt-4o-mini'
    if 'questions' not in st.session_state:
        st.session_state['questions'] = load_questions()
    if 'show_questions' not in st.session_state:
        st.session_state['show_questions'] = False
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'config_set' not in st.session_state:
        st.session_state['config_set'] = False

    # Check if the API key is already set in the environment
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        st.success("OpenAI API key found in environment variables.")
        st.session_state['api_key'] = api_key
        st.session_state['api_key_set'] = True
    elif not st.session_state['api_key_set']:
        # Only ask for the API key if it's not in the environment and not already set
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            st.session_state['api_key'] = api_key
            st.session_state['api_key_set'] = True
            st.success("API key saved for this session.")
            st.rerun()  # This will rerun the script, effectively removing the input field

    if st.session_state['api_key_set']:
        st.success("API key is set for this session.")

    # Model selection
    model_name = st.text_input("Enter the GPT model name", value=st.session_state['model_name'])
    
    if st.button("Set Configuration"):
        if model_name:
            st.session_state['model_name'] = model_name
            st.success(f"Configuration set: Using model {model_name}")
        else:
            st.error("Please provide a model name.")

    st.warning("Your API key is not stored permanently and will only be used for this session.")
    
    # Questions section
    st.subheader("Questions")
    if st.button("Hide Questions" if st.session_state['show_questions'] else "Show Questions"):
        st.session_state['show_questions'] = not st.session_state['show_questions']
        st.rerun()

    # Display and customize questions
    if st.session_state['show_questions']:
        display_and_customize_questions()

    # File upload section
    st.subheader("File Upload")
    st.write("Upload your audio or text file")
    
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "txt", "rtf", "doc", "docx", "pdf", "odt", "md"])
    
    if file is not None:
        st.session_state['uploaded_file'] = file

    if st.session_state['uploaded_file'] is not None:
        file = st.session_state['uploaded_file']
        file_type = detect_file_type(file)
        
        if file_type is None:
            st.error("Unsupported file type. Please upload an audio or text file.")
        else:
            st.write(f"File uploaded: {file.name} (Detected as: {file_type})")
            
            if st.button("Analyze"):
                if file_type == "Audio":
                    st.write(f"Processing audio file: {file.name}")
                    transcription = transcribe_audio(file, st.session_state['api_key'])
                else:
                    st.write(f"Processing text file: {file.name}")
                    transcription = process_file(file)
                
                # Reset analysis results
                st.session_state['analysis_results'] = None
                
                # Process and display results
                process_and_display_results(transcription, st.session_state['questions'])
            
            # Add this condition to display results if they exist
            elif st.session_state['analysis_results'] is not None:
                process_and_display_results(None, None)

    # Initialize session state for changelog visibility
    if 'show_changelog' not in st.session_state:
        st.session_state['show_changelog'] = False

    # Add version number and changelog at the bottom
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: gray;'>Version {VERSION}</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Created by GenAI Jake</p>", unsafe_allow_html=True)
    
    # Toggle changelog visibility
    if st.button("Hide Changelog" if st.session_state['show_changelog'] else "View Changelog"):
        st.session_state['show_changelog'] = not st.session_state['show_changelog']
        st.rerun()

    # Display changelog if show_changelog is True
    if st.session_state['show_changelog']:
        display_changelog()

def display_and_customize_questions():
    st.subheader("Review and Customize Questions")
    
    for category_index, category in enumerate(st.session_state['questions']["project_questions"]):
        st.markdown(f"### {category['category']}")
        for i, question in enumerate(category["questions"]):
            with st.container():
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    q_text = question["text"] if isinstance(question, dict) else question
                    st.text_input(
                        f"Question {i+1}",
                        value=q_text,
                        key=f"q_{category_index}_{i}",
                        on_change=update_question,
                        args=(category_index, i),
                        label_visibility="collapsed"
                    )
                with col2:
                    if st.button("📝", key=f"instruction_{category_index}_{i}", help="Add instructions"):
                        st.session_state[f'show_instruction_{category_index}_{i}'] = True
                with col3:
                    if st.button("🗑️", key=f"delete_{category_index}_{i}", help="Delete question"):
                        delete_question(category_index, i)
                        st.rerun()
                
                if st.session_state.get(f'show_instruction_{category_index}_{i}', False):
                    with st.expander("Question Instructions", expanded=True):
                        instruction = st.text_area(
                            "Instructions",
                            value=question["instruction"] if isinstance(question, dict) and "instruction" in question else "",
                            key=f"instruction_text_{category_index}_{i}",
                            height=100,
                            label_visibility="visible"
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save", key=f"save_instruction_{category_index}_{i}"):
                                save_instruction(category_index, i, instruction)
                                st.success("Instructions saved!")
                                st.session_state[f'show_instruction_{category_index}_{i}'] = False
                                st.rerun()
                        with col2:
                            if st.button("Cancel", key=f"cancel_instruction_{category_index}_{i}"):
                                st.session_state[f'show_instruction_{category_index}_{i}'] = False
                                st.rerun()
        
        # Add question button at the end of each category
        if st.button("Add Question", key=f"add_question_{category_index}"):
            add_question(category_index)
            st.rerun()
        
        st.markdown("---")  # Add a separator after each category

def save_instruction(category_index, question_index, instruction):
    questions = st.session_state['questions']["project_questions"][category_index]["questions"]
    if isinstance(questions[question_index], str):
        # Convert string to dictionary
        questions[question_index] = {
            "text": questions[question_index],
            "instruction": instruction
        }
    else:
        # Update existing dictionary
        questions[question_index]["instruction"] = instruction
    
    save_questions_to_file(st.session_state['questions'])

def save_questions_to_file(questions):
    with open('Questions.json', 'w') as f:
        json.dump(questions, f, indent=2)

def load_questions():
    try:
        with open('Questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Questions.json file not found. Please ensure it exists in the same directory as the script.")
        return {"project_questions": []}

def update_question(category_index, question_index):
    new_text = st.session_state[f"q_{category_index}_{question_index}"]
    st.session_state['questions']["project_questions"][category_index]["questions"][question_index]["text"] = new_text
    save_questions_to_file(st.session_state['questions'])

def add_question(category_index):
    new_question = {"text": "New question", "instruction": ""}
    st.session_state['questions']["project_questions"][category_index]["questions"].append(new_question)
    save_questions_to_file(st.session_state['questions'])

def delete_question(category_index, question_index):
    del st.session_state['questions']["project_questions"][category_index]["questions"][question_index]
    save_questions_to_file(st.session_state['questions'])

def process_and_display_results(transcription, questions):
    if st.session_state['analysis_results'] is None:
        with st.spinner(f"Analyzing content using {st.session_state['model_name']}..."):
            # Check if transcription is a string or an object with a 'text' attribute
            if isinstance(transcription, str):
                text_to_process = transcription
            elif hasattr(transcription, 'text'):
                text_to_process = transcription.text
            else:
                st.error("Invalid transcription format")
                return

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_questions = sum(len(category["questions"]) for category in questions["project_questions"])
            processed_questions = 0

            for category, answer in process_transcription(text_to_process, questions, st.session_state['api_key'], st.session_state['model_name']):
                if not any(result['category'] == category for result in results):
                    results.append({"category": category, "answers": []})
                for result in results:
                    if result['category'] == category:
                        result['answers'].append(answer)
                
                processed_questions += 1
                progress = processed_questions / total_questions
                progress_bar.progress(progress)
                status_text.text(f"Processing: {category} - {answer['question']}")

                # Display the latest result
                st.subheader(category)
                st.markdown(f"**Q: {answer['question']}**")
                st.write(f"A: {answer['answer']}")
                st.markdown("---")

            progress_bar.empty()
            status_text.empty()

        st.session_state['analysis_results'] = results
        if hasattr(transcription, 'text'):
            st.session_state['transcription'] = format_transcript(transcription)
        else:
            st.session_state['transcription'] = transcription  # Store the original text if it's not a transcription object
    else:
        results = st.session_state['analysis_results']
    
    st.subheader("Analysis Results")
    
    # Generate results in different formats
    text_results = generate_text_results(results)
    pdf_results = generate_pdf_results(results)
    docx_results = generate_docx_results(results)
    
    # Create download buttons at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            label="Download as TXT",
            data=text_results,
            file_name="analysis_results.txt",
            mime="text/plain",
            key="download_txt"
        )
    with col2:
        st.download_button(
            label="Download as PDF",
            data=pdf_results,
            file_name="analysis_results.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
    with col3:
        st.download_button(
            label="Download as DOCX",
            data=docx_results,
            file_name="analysis_results.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_docx"
        )
    with col4:
        if 'transcription' in st.session_state and st.session_state['transcription']:
            st.download_button(
                label="Download Transcript",
                data=st.session_state['transcription'],
                file_name="original_transcript.txt",
                mime="text/plain",
                key="download_transcript"
            )
    
    # Display results
    display_results(results)

def transcribe_audio(file, api_key):
    # Save the uploaded file temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(file.getvalue())
    
    # Load the Whisper model
    model = whisper.load_model("base")
    
    # Transcribe the audio
    result = model.transcribe("temp_audio.mp3")
    
    # Remove the temporary file
    os.remove("temp_audio.mp3")
    
    return result

def process_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension in ['.txt', '.md']:
        return file.getvalue().decode('utf-8')
    elif file_extension == '.pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in pdf_reader.pages])
    elif file_extension in ['.doc', '.docx']:
        doc = Document(file)
        return ' '.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def format_transcript(transcription):
    formatted_text = ""
    for segment in transcription['segments']:
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        formatted_text += f"[{start_time} - {end_time}] {segment['text']}\n"
    return formatted_text

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_transcription(text, questions, api_key, model_name):
    client = OpenAI(api_key=api_key)
    
    for category in questions["project_questions"]:
        category_results = {"category": category["category"], "answers": []}
        for question in category["questions"]:
            if isinstance(question, dict):
                question_text = question['text']
            else:
                question_text = question
            prompt = f"Based on the following text, answer this question: {question_text}\n\nText: {text}"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            answer = {
                "question": question_text,
                "answer": response.choices[0].message.content.strip()
            }
            category_results["answers"].append(answer)
            yield category_results["category"], answer

    return

def generate_text_results(results):
    text_output = ""
    for category in results:
        text_output += f"{category['category']}\n\n"
        for qa in category['answers']:
            text_output += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        text_output += "\n"
    return text_output

def generate_pdf_results(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for category in results:
        story.append(Paragraph(category['category'], styles['Heading1']))
        for qa in category['answers']:
            story.append(Paragraph(f"Q: {qa['question']}", styles['Heading3']))
            story.append(Paragraph(f"A: {qa['answer']}", styles['BodyText']))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_docx_results(results):
    doc = Document()
    
    # Define styles
    styles = doc.styles
    heading1_style = styles['Heading 1']
    heading3_style = styles['Heading 3']
    normal_style = styles['Normal']

    for category in results:
        heading = doc.add_paragraph(category['category'])
        heading.style = heading1_style
        for qa in category['answers']:
            question = doc.add_paragraph(f"Q: {qa['question']}")
            question.style = heading3_style
            answer = doc.add_paragraph(f"A: {qa['answer']}")
            answer.style = normal_style
            doc.add_paragraph()  # Add a blank line

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def display_results(results):
    for category in results:
        st.subheader(category['category'])
        for qa in category['answers']:
            st.markdown(f"**Q: {qa['question']}**")
            st.write(f"A: {qa['answer']}")
            st.markdown("---")

def display_changelog():
    try:
        with open('CHANGELOG.md', 'r') as f:
            changelog = f.read()
        st.markdown(changelog)
    except FileNotFoundError:
        st.error("CHANGELOG.md file not found.")

if __name__ == "__main__":
    main()
