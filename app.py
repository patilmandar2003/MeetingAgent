import os
import json
import subprocess
import tempfile
from typing import List, TypedDict, Optional
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import asyncio
import streamlit as st
import whisper
import torch

# Define MeetingAgent state
class MeetingAgent(TypedDict, total=False):
    agenda: Optional[str]
    tasks: Optional[str]
    deadline: Optional[str]
    meetingFile: Optional[str]
    fileData: Optional[str]
    summary: Optional[str]

# Clear GPU cache
torch.cuda.empty_cache()

# Initialize model
model = OllamaLLM(model='phi3:mini')

# Initialize audio to text model
@st.cache_resource
def load_whisper():
    return whisper.load_model("small")

whisper_model = load_whisper()

# Setting Streamlit UI
st.set_page_config(page_title="Meeting Agent", page_icon=":robot_face:", layout="centered")
st.title("Meeting Agent 🤖")

st.write("Upload your meeting file (text, pdf, audio) and get a concise summary of discussion and tasks.")

# File upload
uploaded_file = st.file_uploader("Upload meeting File", 
                               type=['txt', 'pdf', 'mp3', 'wav', 'm4a', 'mp4', 'mov', 'avi'])

# Save transcript of meeting
def save_transcript(meeting_id, transcript_text):
    os.makedirs("data/transcripts", exist_ok=True)
    with open(f"data/transcripts/{meeting_id}.json", "w") as f:
        json.dump({"id": meeting_id, "text": transcript_text}, f)

def DataCapture(state: MeetingAgent):
    # print("File selected")
    # upload_file = "./meeting_transcript.pdf"
    # # upload_file = st.file_uploader("Upload meeting File", type=['txt', 'pdf', 'mp3', 'wav', 'm4a'])
    
    # return {'meetingFile': upload_file}

    if uploaded_file:
        # Save file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        return {'meetingFile': temp_path}
    else:
        st.warning("Please upload a file to continue.")
        st.stop()

def SortFile(state: MeetingAgent):
    # print("Sorting File")
    upload_file = state['meetingFile']

    if upload_file.endswith('.txt'):
        return "textFile"
    elif upload_file.endswith('.pdf'):
        return "pdfFile"
    else:
        return "whisper"

def ProcessTextFile(state: MeetingAgent):
    # print("Processing Text file")
    file_data = state['meetingFile']

    try:
        with open(file_data, "r") as file:
            content = file.read()
            # print(content)
    # except FileNotFoundError:
    #     print("Error: The file 'your_file_name.txt' was not found.")
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        st.stop()

    return {'fileData': content}

def ProcessPDFFile(state: MeetingAgent):
    # print("Processing pdf file")
    file= PyPDFLoader((state['meetingFile']))

    async def load_pages(loader: PyPDFLoader) -> List[Document]:
        pages = []

        async for page in loader.alazy_load():
            pages.append(page)

            return pages

    meeting_pages = asyncio.run(load_pages(file))
    combined_text = "\n".join([page.page_content for page in meeting_pages])

    return {'fileData': combined_text}

def ProcessAVFile(state: MeetingAgent):
    # print("Processing AV File")
    file = state['meetingFile']

    st.info("🎧 Processing audio/video file...")

    # Step 1. Validate file existence
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        st.error("Uploaded audio/video file is empty or missing.")
        st.stop()

    # Step 2. Handle file type
    file_ext = os.path.splitext(file)[1].lower()
    audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

    # Step 3. Extract audio if it's a video
    audio_path = file
    temp_wav = None

    if file_ext not in audio_extensions:
        # st.write("🎬 Detected video file — extracting audio using FFmpeg...")
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        command = [
            "ffmpeg", "-y", "-i", file,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            temp_wav, "-loglevel", "error"
        ]

        try:
            subprocess.run(command, check=True)
            audio_path = temp_wav
        except subprocess.CalledProcessError:
            st.error("❌ FFmpeg failed to extract audio. Make sure FFmpeg is installed.")
            st.stop()

    # Step 4. Transcribe safely with Whisper
    with st.spinner("🔊 Transcribing with Whisper..."):
        try:
            transcription = whisper_model.transcribe(audio_path, fp16=False)
            if not transcription or 'text' not in transcription:
                raise ValueError("Whisper returned empty text.")
            return {'fileData': transcription["text"]}
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
        finally:
            # Clean up temp file if it was created
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)


    return {'fileData': transcription}

def SummarizeMeeting(state: MeetingAgent):
    # print("Summarizing meeting")
    st.subheader("🔍 Summarizing Meeting...")
    data = state['fileData']

    prompt = f"""
    Summarize the following meeting transcript:
    {data}

    Provide a concise summary divided into:
    1. Key Discussions
    2. Tasks and Deadlines
    3. Important Decisions
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    print(response)
    st.success("✅ Summary generated!")
    st.markdown("### 📝 Meeting Summary:")
    st.write(response)

    # Save summary of meeting
    save_transcript("meeting_summary", response)

    return {'summary': response}

def AskMeetingQuestions(state: MeetingAgent):
    st.subheader("❓ Ask Questions About the Meeting...")
    
    # Persistent chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show chat messages
    for message in st.session_state.chat_history:
        role = "👤 You" if isinstance(message, HumanMessage) else "🤖 Assistant"
        st.markdown(f"**{role}:** {message.content}")

    # Input box — pressing Enter automatically triggers rerun
    question = st.text_input("Ask a question about the meeting:", key="user_input")

    if question:
        # Append user message
        st.session_state.chat_history.append(HumanMessage(content=question))

        # Get response from model
        response = model.invoke([HumanMessage(content=question)])
        answer = response.content if hasattr(response, "content") else str(response)

        # Append AI response
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Save the interaction
        save_transcript(f"meeting_question_{question}", answer)

        # Clear input box
        st.session_state.user_input = ""

        # Trigger rerun to show updated messages
        st.rerun()

    return {}

# Create graph
meeting_graph = StateGraph(MeetingAgent)

# Add nodes
meeting_graph.add_node('DataCapture', DataCapture)
# meeting_graph.add_node('SortFile', SortFile)
meeting_graph.add_node('ProcessTextFile', ProcessTextFile)
meeting_graph.add_node('ProcessPDFFile', ProcessPDFFile)
meeting_graph.add_node('ProcessAVFile', ProcessAVFile)
meeting_graph.add_node('SummarizeMeeting', SummarizeMeeting)
meeting_graph.add_node('AskMeetingQuestions', AskMeetingQuestions)

# Add edges
meeting_graph.add_edge(START, 'DataCapture')
meeting_graph.add_conditional_edges('DataCapture', 
                                    SortFile,
                                   {
                                   'textFile': 'ProcessTextFile',
                                   'pdfFile': 'ProcessPDFFile',
                                   'whisper': 'ProcessAVFile'
                                   }
)
meeting_graph.add_edge('ProcessTextFile', 'SummarizeMeeting')
meeting_graph.add_edge('ProcessPDFFile', 'SummarizeMeeting')
meeting_graph.add_edge('ProcessAVFile', 'SummarizeMeeting')
meeting_graph.add_edge('SummarizeMeeting', "AskMeetingQuestions")
meeting_graph.add_edge('AskMeetingQuestions', END)

# Compile graph
compiled_graph = meeting_graph.compile()

# meeting_agent = compiled_graph.invoke({})

# Run when file is uploaded
if uploaded_file is not None:
    if st.button("🚀 Process and Summarize Meeting"):
        with st.spinner("Processing..."):
            compiled_graph.invoke({})