import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request,Query,APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from twilio.rest import Client
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from scipy.spatial.distance import cdist
import numpy as np
import logging
import time

load_dotenv()

OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
NGROKURL = os.getenv("NGROKURL")

VOICE = 'alloy'


SYSTEM_MESSAGE = (
    "<bio> You are an AI Agent named 'Dhvani (Digital Helper for Voice Assisted Natural Interaction)' developed by Varun Yadgude.</bio>"
    "<voice_type>Soft-spoken but confident woman with an Indian accent.</voice_type>"
    "<voice_personality>ALWAYS USE FILLER WORDS, stay positive.</voice_personality>"
    "<voice_speed>MODERATE</voice_speed>"
    "<task>You are tasked with answering user queries by retrieving information from a PDF document stored in Chroma DB, using a Retrieval-Augmented Generation (RAG) system.</task>"
    "<important_rules>ALWAYS SPEAK IN THE LANGUAGE THE USER IS TALKING IN. ONLY ASK QUESTIONS ONCE. NEVER SAY 'hi, how can I help you?' TWICE.</important_rules>"
    "<instructions>You are assisting users by providing answers to their queries based on information extracted from PDF files:"
    "1. When a user asks a question, search the Chroma DB for relevant information from the PDF."
    "2. Use the 'get_additional_context' function to retrieve information related to the user query from the Chroma DB."
    "3. If relevant information is found, provide a concise and accurate answer."
    "4. If no relevant information is found, politely apologize and inform the user that you couldn't find an answer."
    "5. Never provide any information outside of what is retrieved from Chroma DB."
    "</instructions>"
    "<goal>Provide accurate and concise responses based on the information retrieved from the Chroma DB using the RAG system.</goal>"
    "<additional_instructions>"
    "1. Use the 'get_additional_context' function for all user queries to retrieve information from the Chroma DB."
    "2. Structure the function call as follows: "
    "   - Function call to 'get_additional_context' with arguments where the query must start with: 'A user asked: [include the exact transcription of the user's request]'."
    "   - Enhance the user query by adding depth, specificity, and clarity without altering its intent."
    "   - Tailor the information as if the user were asking an expert in the relevant field, ensuring the request is comprehensive."
    "3. If 'get_additional_context' responds with 'sorry,' apologize politely and do not provide any speculative answers."
    "4. Avoid using any internal knowledge base outside of the Chroma DB."
    "</additional_instructions>"
    
    "<response_guidelines>"
    "1. You can answer everything the user asked via 'get_additional_context,' even regarding individuals or personal questions if the context allows."
    "2. Do not say anything before calling 'get_additional_context.'"
    "3. Use ONLY information retrieved from 'get_additional_context' for your responses."
    "4. Keep responses under 50 words unless absolutely necessary."
    "5. Never justify or explain your answers."
    "6. Never mention the 'get_additional_context' process or the RAG system in your responses."
    "7. Never repeat user queries verbatim."
    "8. Never mention anything regarding user queries from your internal knowledge base."
    "</response_guidelines>"
)


LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created','response.audio.done','response.funtion_call_arguments.done',
    'conversation.item.input_audio_transcription.completed'
]
SHOW_TIMING_MATH = False

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN

client = Client(account_sid, auth_token)
my_number = TWILIO_PHONE_NUMBER
base_url =NGROKURL

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

chroma_client = chromadb.PersistentClient()

collection_name = "vdb_collection"
try:
    chroma_client.delete_collection(name=collection_name)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

loader = PyMuPDFLoader("./LPG.pdf")
pages = loader.load()

document = ""
for page in pages:
    document += page.page_content

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=800,
    chunk_overlap=400,
)
chunks = text_splitter.split_text(document)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

for i, chunk in enumerate(chunks):
    embedding = embeddings.embed_query(chunk)
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )
print("PDF chunks successfully loaded into Chroma DB.")

@router.get("/LPG_Dhvani")
def make_call(to: str = Query(...)):
    call = client.calls.create(
        to = to,  
        from_ = my_number,  
        url=f'{base_url}/incomming-call' 
    )
    return {"message": f"Call initiated with SID: {call.sid}"}


@router.api_route("/incomming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    response.say("Welcome to EasyAI")
    response.pause(length=1)
    response.say("O.K. you can start talking!")
    host = request.url.hostname
    print("host :",host)
    connect = Connect()
    connect.stream(url=f'wss://{host}/lpg/media-stream')
    response.append(connect)
    print("response :",str(response))
    return HTMLResponse(content=str(response), media_type="application/xml")

@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    try:   
        print("Client connected")
        await websocket.accept()
        try:
            async with websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
                additional_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as openai_ws:
                await pdf_initialize_session(openai_ws)

                stream_sid = None
                latest_media_timestamp = 0
                last_assistant_item = None
                mark_queue = []
                response_start_timestamp_twilio = None

                async def receive_from_twilio():
                    nonlocal stream_sid, latest_media_timestamp
                    try:
                        async for message in websocket.iter_text():
                            data = json.loads(message)
                            if data['event'] == 'media' and openai_ws.connection_made:
                                latest_media_timestamp = int(data['media']['timestamp'])
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": data['media']['payload']
                                }
                                await openai_ws.send(json.dumps(audio_append))
                            elif data['event'] == 'start':
                                stream_sid = data['start']['streamSid']
                                print(f"Incoming stream has started {stream_sid}")
                                response_start_timestamp_twilio = None
                                latest_media_timestamp = 0
                                last_assistant_item = None
                            elif data['event'] == 'mark':
                                if mark_queue:
                                    mark_queue.pop(0)
                    except WebSocketDisconnect:
                        print("Client disconnected.")
                        if openai_ws.open:
                            await openai_ws.close()
                        if openai_ws.open:
                            await openai_ws.close()

                async def send_to_twilio():
                    nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
                    try:
                        async for openai_message in openai_ws:
                            response = json.loads(openai_message)

                            if response.get('type') == 'response.audio.delta' and 'delta' in response:
                                audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(audio_delta)

                                if response_start_timestamp_twilio is None:
                                    response_start_timestamp_twilio = latest_media_timestamp

                                if response.get('item_id'):
                                    last_assistant_item = response['item_id']

                                await send_mark(websocket, stream_sid)

                            if response.get('type') == 'input_audio_buffer.speech_started':
                                if last_assistant_item:
                                    await handle_speech_started_event()

                            if response['type'] == 'response.function_call_arguments.done':
                                try:
                                    function_name = response['name']
                                    call_id = response['call_id']
                                    arguments = json.loads(response['arguments'])

                                    if function_name == 'retrieve_from_chroma':
                                        logger.info("Retrieving additional context from ChromaDB")

                                        start_time = time.time()

                                        retrieved_chunks = retrieve_from_chroma(
                                            query=arguments['query'], 
                                            client=chroma_client, 
                                            collection_name="vdb_collection", 
                                            embeddings=embeddings
                                        )

                                        logger.info("Additional context retrieved from ChromaDB")
                                        logger.debug(f"Retrieved context: {retrieved_chunks}")

                                        end_time = time.time()
                                        elapsed_time = end_time - start_time
                                        logger.info(f"get_additional_context execution time: {elapsed_time:.4f} seconds")

                                        custom_prompt = f"""
                                        The following relevant information was retrieved from the knowledge base:
                                        {retrieved_chunks}

                                        User's query: {arguments['query']}
                                        Please generate a concise and precise response based on the provided information.
                                        """

                                        function_response = {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": custom_prompt
                                            }
                                        }
                                        await openai_ws.send(json.dumps(function_response))

                                        await openai_ws.send(json.dumps({"type": "response.create"}))

                                except Exception as e:
                                    print(f"Error handling function call: {e}")

                    except Exception as e:
                        print(f"Error in send_to_twilio: {e}")

                async def handle_speech_started_event():
                    nonlocal response_start_timestamp_twilio, last_assistant_item
                    print("Handling speech started event.")
                    if mark_queue and response_start_timestamp_twilio is not None:
                        elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                        if SHOW_TIMING_MATH:
                            print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                        if last_assistant_item:
                            if SHOW_TIMING_MATH:
                                print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                            truncate_event = {
                                "type": "conversation.item.truncate",
                                "item_id": last_assistant_item,
                                "content_index": 0,
                                "audio_end_ms": elapsed_time
                            }
                            await openai_ws.send(json.dumps(truncate_event))

                        await websocket.send_json({
                            "event": "clear",
                            "streamSid": stream_sid
                        })

                        mark_queue.clear()
                        last_assistant_item = None
                        response_start_timestamp_twilio = None

                async def send_mark(connection, stream_sid):
                    if stream_sid:
                        mark_event = {
                            "event": "mark",
                            "streamSid": stream_sid,
                        }
                        await connection.send_json(mark_event)
                        mark_queue.append('mark_event')

                await asyncio.gather(receive_from_twilio(), send_to_twilio())
        except WebSocketDisconnect:
            print("WebSocket disconnected 2")
        except Exception as e:
            print(f"Error: {e}")        
    except WebSocketDisconnect:
        print("WebSocket Disconnected 1")

async def send_initial_conversation_item(openai_ws):
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


logger = logging.getLogger(__name__)

def retrieve_from_chroma(query, client, collection_name, embeddings, top_k=3):

    tries = 0
    max_retries = 2
    while tries <= max_retries:
        try:
            query_embedding = embeddings.embed_query(query)

            collection = client.get_collection(name=collection_name)

            docs = collection.get(include=["documents", "embeddings"])
            documents = docs["documents"]
            document_embeddings = np.array(docs["embeddings"])

            similarities = 1 - cdist([query_embedding], document_embeddings, metric="cosine")[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_chunks = [documents[idx] for idx in top_indices]

            response = "\n\n".join(top_chunks)
            logger.info(f"ChromaDB response generated: {response}")
            return response
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB::Try {tries}::Error: {e}")
            time.sleep(2)
        tries += 1

    return "Sorry, I couldn't retrieve the information. Please try again later."

async def pdf_initialize_session(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "tools": [
                {
                    "type": "function",
                    "name": "retrieve_from_chroma",
                    "description": "Retrieve information from a ChromaDB vector database based on a user's query. The function finds semantically similar results from the database to answer the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's query to be used for searching the ChromaDB vector database. The query should represent the information the user is seeking."
                        }
                        },
                        "required": ["query"]
                    }
                }
            ],
            "tool_choice": "auto"
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))










