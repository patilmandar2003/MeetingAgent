import base64
from typing import List, TypedDict, Annotated, Optional, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pypdf
import asyncio
from IPython.display import Image, display
import re
import streamlit as st
import whisper

class MeetingAgent(TypedDict, total=False):
    agenda: Optional[str]
    tasks: Optional[str]
    deadline: Optional[str]
    meetingFile: Optional[str]
    fileData: Optional[str]

# Initialize model
model = OllamaLLM(model='phi3:mini')

# Initialize audio to text model
whisper_model = whisper.load_model("small")

# Setting Streamlit UI
st.set_page_config(page_title="Meeting Agent", page_icon=":robot_face:", layout="centered")
st.title("Meeting Agent 🤖")

st.write("Upload your meeting file (text, pdf, audio) and get a concise summary of discussion and tasks.")

# File upload
uploaded_file = st.file_uploader("Upload meeting File", 
                               type=['txt', 'pdf', 'mp3', 'wav', 'm4a'])

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

    # transcription = model.transcribe(file)
    with st.spinner("Transcribing audio/video using Whisper..."):
        transcription = whisper_model.transcribe(file)["text"]

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
meeting_graph.add_edge('SummarizeMeeting', END)

# Compile graph
compiled_graph = meeting_graph.compile()

# meeting_agent = compiled_graph.invoke({})

# Run when file is uploaded
if uploaded_file is not None:
    if st.button("🚀 Process and Summarize Meeting"):
        with st.spinner("Processing..."):
            compiled_graph.invoke({})