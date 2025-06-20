import os
import re
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wavio  # To save the audio file
# Third-party imports
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import tiktoken
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv
import time
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoContent:
    """Data class to store video content information"""
    video_id: str
    title: str
    transcript: str
    chunks: List[str]
    language: str = 'en'
    duration: Optional[float] = None

class YouTubeProcessor:
    """Handles YouTube video processing, transcript extraction, and audio transcription"""
    
    # Supported languages for transcript extraction
    SUPPORTED_LANGUAGES = {
        'en': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
        'es': ['es', 'es-ES', 'es-MX', 'es-AR'],
        'fr': ['fr', 'fr-FR', 'fr-CA'],
        'de': ['de', 'de-DE'],
        'it': ['it', 'it-IT'],
        'pt': ['pt', 'pt-BR', 'pt-PT'],
        'ru': ['ru', 'ru-RU'],
        'ja': ['ja', 'ja-JP'],
        'ko': ['ko', 'ko-KR'],
        'zh': ['zh', 'zh-CN', 'zh-TW', 'zh-HK'],
        'ar': ['ar', 'ar-SA'],
        'hi': ['hi', 'hi-IN'],
        'th': ['th', 'th-TH'],
        'ml': ['ml', 'ml-IN'],
        'vi': ['vi', 'vi-VN'],
        'nl': ['nl', 'nl-NL'],
        'sv': ['sv', 'sv-SE'],
        'no': ['no', 'no-NO'],
        'da': ['da', 'da-DK'],
        'fi': ['fi', 'fi-FI'],
        'tr': ['tr', 'tr-TR'],
        'pl': ['pl', 'pl-PL'],
        'cs': ['cs', 'cs-CZ'],
        'hu': ['hu', 'hu-HU'],
        'ro': ['ro', 'ro-RO'],
        'uk': ['uk', 'uk-UA'],
        'id': ['id', 'id-ID'],
        'ms': ['ms', 'ms-MY'],
        'tl': ['tl', 'tl-PH'],
    }
    
    # Language names for display
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'th': 'Thai',
        'ml': 'Malayalam',
        'vi': 'Vietnamese',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'tr': 'Turkish',
        'pl': 'Polish',
        'cs': 'Czech',
        'hu': 'Hungarian',
        'ro': 'Romanian',
        'uk': 'Ukrainian',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Filipino',
    }
    
    def __init__(self):
        self.genai_client = None
        self._setup_genai()
    
    def _setup_genai(self):
        """Initialize Gemini AI client"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.genai_client = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Gemini AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            raise
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        try:
            parsed_url = urlparse(url)
            
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query)['v'][0]
                elif parsed_url.path.startswith('/embed/'):
                    return parsed_url.path.split('/')[2]
            elif parsed_url.hostname in ['youtu.be']:
                return parsed_url.path[1:]
            
            raise ValueError("Invalid YouTube URL format")
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def get_video_info(self, video_id: str) -> Dict:
        """Get video metadata using yt-dlp"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                
            return {
                'title': info.get('title', 'Unknown Title'),
                'duration': info.get('duration', 0),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'title': 'Unknown Title', 'duration': 0}
    
    def get_transcript(self, video_id: str, preferred_language: str = 'en') -> tuple[Optional[str], str]:
        """Try to get existing transcript from YouTube in preferred language"""
        try:
            # Get all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get transcript in preferred language first
            languages_to_try = self.SUPPORTED_LANGUAGES.get(preferred_language, ['en'])
            
            transcript_text = None
            detected_language = preferred_language
            
            # First, try manually created transcripts in preferred language
            for lang_code in languages_to_try:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang_code])
                    transcript_data = transcript.fetch()
                    transcript_text = ' '.join([item['text'] for item in transcript_data])
                    detected_language = lang_code[:2]  # Get base language code
                    logger.info(f"Found manual transcript in {lang_code} for video {video_id}")
                    break
                except:
                    continue
            
            # If no manual transcript, try auto-generated in preferred language
            if not transcript_text:
                for lang_code in languages_to_try:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang_code])
                        transcript_data = transcript.fetch()
                        transcript_text = ' '.join([item['text'] for item in transcript_data])
                        detected_language = lang_code[:2]
                        logger.info(f"Found auto-generated transcript in {lang_code} for video {video_id}")
                        break
                    except:
                        continue
            
            # If still no transcript, try any available transcript and translate if needed
            if not transcript_text:
                try:
                    # Get first available transcript
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        first_transcript = available_transcripts[0]
                        
                        # Try to translate to preferred language if different
                        if preferred_language != 'en' and preferred_language in self.SUPPORTED_LANGUAGES:
                            try:
                                translated = first_transcript.translate(preferred_language)
                                transcript_data = translated.fetch()
                                transcript_text = ' '.join([item['text'] for item in transcript_data])
                                detected_language = preferred_language
                                logger.info(f"Translated transcript to {preferred_language} for video {video_id}")
                            except:
                                # Fall back to original transcript
                                transcript_data = first_transcript.fetch()
                                transcript_text = ' '.join([item['text'] for item in transcript_data])
                                detected_language = first_transcript.language_code[:2]
                                logger.info(f"Using original transcript in {detected_language} for video {video_id}")
                        else:
                            transcript_data = first_transcript.fetch()
                            transcript_text = ' '.join([item['text'] for item in transcript_data])
                            detected_language = first_transcript.language_code[:2]
                            logger.info(f"Using available transcript in {detected_language} for video {video_id}")
                except:
                    pass
            
            if transcript_text:
                return transcript_text, detected_language
            else:
                logger.info(f"No transcript found for video {video_id}")
                return None, preferred_language
                
        except NoTranscriptFound:
            logger.info(f"No transcript found for video {video_id}")
            return None, preferred_language
        except Exception as e:
            logger.warning(f"Error retrieving transcript: {e}")
            return None, preferred_language
    
    def download_audio(self, video_id: str, output_path: str = "./temp_audio") -> str:
        """Download audio from YouTube video"""
        try:
            os.makedirs(output_path, exist_ok=True)
            audio_file = os.path.join(output_path, f"{video_id}.wav")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_path, f"{video_id}.%(ext)s"),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            logger.info(f"Audio downloaded successfully: {audio_file}")
            return audio_file
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise
    
    def transcribe_audio_with_gemini(self, audio_file_path: str, target_language: str = 'en') -> str:
        """Transcribe audio using Gemini 2.0 Flash with language specification"""
        try:
            # Upload audio file to Gemini
            audio_file = genai.upload_file(audio_file_path)
            
            # Wait for processing
            import time
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise ValueError("Audio processing failed")
            
            # Generate transcript with language specification
            language_name = self.LANGUAGE_NAMES.get(target_language, 'English')
            
            prompt = f"""
            Please transcribe this audio file accurately in {language_name}.
            
            Instructions:
            - Provide a clean transcript without timestamps or speaker labels
            - Focus on capturing the main content and ideas clearly
            - If the audio is in a different language than {language_name}, transcribe it in the original language
            - Maintain the natural flow of speech and proper grammar
            - If there are multiple speakers, separate their speech clearly
            
            Transcript:
            """
            
            response = self.genai_client.generate_content([prompt, audio_file])
            transcript = response.text
            
            # Clean up uploaded file
            genai.delete_file(audio_file.name)
            
            logger.info(f"Audio transcribed successfully with Gemini in {language_name}")
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio with Gemini: {e}")
            raise
    
    def process_video(self, youtube_url: str, preferred_language: str = 'en') -> VideoContent:
        """Main method to process YouTube video and extract content"""
        try:
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            logger.info(f"Processing video ID: {video_id}")
            
            # Get video info
            video_info = self.get_video_info(video_id)
            
            # Try to get existing transcript first
            transcript, detected_language = self.get_transcript(video_id, preferred_language)
            
            # If no transcript, transcribe audio
            if not transcript:
                logger.info("No existing transcript found, downloading and transcribing audio...")
                audio_file = self.download_audio(video_id)
                try:
                    transcript = self.transcribe_audio_with_gemini(audio_file, preferred_language)
                    detected_language = preferred_language
                finally:
                    # Clean up audio file
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                    # Clean up temp directory if empty
                    temp_dir = os.path.dirname(audio_file)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
            
            # Create chunks for RAG
            chunks = self.create_text_chunks(transcript)
            
            return VideoContent(
                video_id=video_id,
                title=video_info['title'],
                transcript=transcript,
                chunks=chunks,
                language=detected_language,
                duration=video_info.get('duration')
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def create_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for RAG"""
        try:
            # Use tiktoken for more accurate token counting
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                start = end - overlap
            
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
            # Fallback to simple splitting
            return self._simple_chunk_split(text, chunk_size, overlap)
    
    def _simple_chunk_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple fallback chunking method"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

class VectorStore:
    """Handles Pinecone vector database operations"""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.embedding_model = None
        self._setup_pinecone()
        self._setup_embeddings()
    
    def _setup_pinecone(self):
        """Initialize Pinecone client"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            self.pc = Pinecone(api_key=api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise


    def check_existing_video(self, video_id: str) -> Optional[VideoContent]:
        """Check if a video exists in the vector database and return its content if found"""
        try:
            if not self.index:
                self.create_index()

            # Query Pinecone for vectors matching the video_id
            query_result = self.index.query(
                vector=[0] * 384,  # Dummy vector since we're filtering by metadata
                top_k=1000,  # Large enough to retrieve all chunks
                include_metadata=True,
                filter={"video_id": video_id}
            )

            # If no matches, return None
            if not query_result['matches']:
                logger.info(f"No existing content found for video {video_id}")
                return None

            # Extract metadata from matches
            chunks = []
            metadata = query_result['matches'][0]['metadata']  # Use first match for general metadata
            title = metadata.get('video_title', 'Unknown Title')
            language = metadata.get('language', 'en')
            duration = None  # Duration might not be stored in metadata; set to None

            # Collect chunks in order based on chunk_index
            chunk_data = sorted(
                [(match['metadata']['chunk_index'], match['metadata']['text']) for match in query_result['matches']],
                key=lambda x: x[0]
            )
            chunks = [chunk[1] for chunk in chunk_data]

            # Reconstruct transcript by joining chunks (removing overlap if possible)
            transcript = ' '.join(chunks)  # Simple join; could be refined to handle overlap

            logger.info(f"Reconstructed content for video {video_id} with {len(chunks)} chunks")
            return VideoContent(
                video_id=video_id,
                title=title,
                transcript=transcript,
                chunks=chunks,
                language=language,
                duration=duration
            )

        except Exception as e:
            logger.error(f"Error checking existing video content: {e}")
            return None
    
    def _setup_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def create_index(self, index_name: str = "youtube-qa-bot", dimension: int = 384):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new index: {index_name}")
                
                # Wait for index to be ready
                import time
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            else:
                logger.info(f"Using existing index: {index_name}")
            
            self.index = self.pc.Index(index_name)
            return self.index
            
        except Exception as e:
            logger.error(f"Error with Pinecone index: {e}")
            raise
    
    def store_video_content(self, video_content: VideoContent):
        """Store video content chunks in vector database"""
        try:
            if not self.index:
                self.create_index()
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(video_content.chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(video_content.chunks, embeddings)):
                vector_id = f"{video_content.video_id}_chunk_{i}"
                
                metadata = {
                    "video_id": video_content.video_id,
                    "video_title": video_content.title,
                    "chunk_index": i,
                    "text": chunk,
                    "chunk_length": len(chunk),
                    "language": video_content.language
                }
                
                vectors.append((vector_id, embedding.tolist(), metadata))
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors)} vectors for video {video_content.video_id}")
            
        except Exception as e:
            logger.error(f"Error storing video content: {e}")
            raise
    
    def search_similar_content(self, query: str, video_id: str, top_k: int = 5) -> List[Dict]:
        """Search for similar content in the vector database"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search with filter for specific video
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter={"video_id": video_id}
            )
            
            # Extract relevant information
            similar_chunks = []
            for match in results['matches']:
                similar_chunks.append({
                    'text': match['metadata']['text'],
                    'score': match['score'],
                    'chunk_index': match['metadata']['chunk_index']
                })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            raise

class QAChatbot:
    """Main chatbot class that handles Q&A using RAG"""
    
    def __init__(self):
        self.youtube_processor = YouTubeProcessor()
        self.vector_store = VectorStore()
        self.conversation_history = []
    
    def load_video(self, youtube_url: str, preferred_language: str = 'en') -> VideoContent:
        """Load and process a YouTube video, checking if it's already in the vector database"""
        try:
            # Extract video ID
            video_id = self.youtube_processor.extract_video_id(youtube_url)
            logger.info(f"Checking if video {video_id} exists in vector database")

            # Check if video content already exists in the vector store
            existing_content = self.vector_store.check_existing_video(video_id)

            if existing_content:
                logger.info(f"Video {video_id} found in vector database, reusing existing content")
                return existing_content

            # If not found, process the video
            logger.info(f"Video {video_id} not found in vector database, processing new content")
            video_content = self.youtube_processor.process_video(youtube_url, preferred_language)

            # Store in vector database
            self.vector_store.store_video_content(video_content)

            logger.info(f"Successfully loaded and stored video: {video_content.title} (Language: {video_content.language})")
            return video_content

        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise
    def answer_question(self, question: str, video_id: str, video_title: str = "", video_language: str = 'en') -> str:
        """Answer a question about the video content using RAG"""
        try:
            # Search for relevant content
            similar_chunks = self.vector_store.search_similar_content(question, video_id)
            
            if not similar_chunks:
                return "I couldn't find relevant information in the video to answer your question."
            
            # Prepare context from similar chunks
            context = "\n\n".join([chunk['text'] for chunk in similar_chunks[:3]])
            
            # Get language name for prompt
            language_name = self.youtube_processor.LANGUAGE_NAMES.get(video_language, 'English')
            
            # Create multilingual prompt for Gemini
            prompt = f"""
            You are a helpful assistant that answers questions about YouTube video content.
            The video content is in {language_name}, and you should respond in the same language as the question.
            
            Video Title: {video_title}
            Video Language: {language_name}
            
            Context from the video:
            {context}
            
            Question: {question}
            
            Instructions:
            - Answer the question based on the provided context from the video
            - Respond in the same language as the question when possible
            - If the question is in English but the video is in another language, provide the answer in English but mention the original language
            - Be specific and reference details from the video when possible
            - If the context doesn't contain enough information to answer the question, say so
            - Keep your answer concise but informative
            - Don't make up information that's not in the context
            - If there are language barriers, do your best to bridge them clearly
            
            Answer:
            """
            
            # Generate response using Gemini
            response = self.youtube_processor.genai_client.generate_content(prompt)
            answer = response.text
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'video_id': video_id,
                'language': video_language
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "I encountered an error while trying to answer your question. Please try again."


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio using SpeechRecognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            # Use Google Web Speech API for transcription
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            logger.error("Google Web Speech API could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Web Speech API; {e}")
            return ""

def record_audio(duration: int, output_file: str):
    """Record audio from the microphone and save it to a file."""
    logger.info("Recording audio...")
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(output_file, recording, fs, sampwidth=2)  # Save as WAV file
    logger.info("Audio recording complete.")

# Streamlit Web Interface
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="YouTube Q&A Chatbot",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 YouTube Q&A Chatbot with RAG")
    st.markdown("Ask questions about any YouTube video content!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = QAChatbot()
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for video loading
    with st.sidebar:
        st.header("📹 Load Video")
        youtube_url = st.text_input("Enter YouTube URL:")
        
        # Language selection
        language_options = {
            'en': '🇺🇸 English',
            'es': '🇪🇸 Spanish',
            'fr': '🇫🇷 French',
            'de': '🇩🇪 German',
            'it': '🇮🇹 Italian',
            'pt': '🇵🇹 Portuguese',
            'ru': '🇷🇺 Russian',
            'ja': '🇯🇵 Japanese',
            'ko': '🇰🇷 Korean',
            'zh': '🇨🇳 Chinese',
            'ar': '🇸🇦 Arabic',
            'hi': '🇮🇳 Hindi',
            'th': '🇹🇭 Thai',
            'ml': '🇮🇳 Malayalam',
            'vi': '🇻🇳 Vietnamese',
            'nl': '🇳🇱 Dutch',
            'tr': '🇹🇷 Turkish',
            'pl': '🇵🇱 Polish',
            'id': '🇮🇩 Indonesian',
        }
        
        selected_language = st.selectbox(
            "Preferred Language:",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0
        )
        
        if st.button("Process Video"):
            if youtube_url:
                try:
                    with st.spinner("Processing video... This may take a few minutes."):
                        video_content = st.session_state.chatbot.load_video(youtube_url, selected_language)
                        st.session_state.current_video = video_content
                        st.session_state.messages = []  # Clear previous messages
                    
                    st.success(f"Video loaded successfully!")
                    st.write(f"**Title:** {video_content.title}")
                    st.write(f"**Language:** {language_options.get(video_content.language, video_content.language)}")
                    st.write(f"**Duration:** {video_content.duration/60:.1f} minutes" if video_content.duration else "Duration: Unknown")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
            else:
                st.warning("Please enter a YouTube URL")
        
        # Display current video info
        if st.session_state.current_video:
            st.markdown("---")
            st.markdown("**Current Video:**")
            st.write(st.session_state.current_video.title)
            st.write(f"Language: {language_options.get(st.session_state.current_video.language, st.session_state.current_video.language)}")

            # Clear chat button
            st.markdown("---")
            if st.button("🧹 Clear Chat", key="clear_chat"):
                st.session_state.messages = []
                st.success("Chat history cleared!")

    # Main chat interface
    if st.session_state.current_video:
        st.markdown(f"### Chatting about: {st.session_state.current_video.title}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        prompt = st.chat_input("Ask a question about the video...")
        
        st.markdown("---")  # Add a separator

        # Record Voice Question button          

        def threaded_record(output_file: str, duration: int = 5):
            """Wrapper to run the audio recording in a separate thread"""
            record_audio(duration=duration, output_file=output_file)

        # Layout: button and dynamic status text
        col1, col2 = st.columns([1, 5])

        with col1:
            record_clicked = st.button("🎙️ Record", key="record_button")

        with col2:
            status_placeholder = st.empty()
            status_placeholder.markdown("⬅️ Click here to start recording")

        # Start recording if button clicked
        if record_clicked:
            audio_file_path = "temp_audio.wav"

            # Start recording in background thread
            thread = threading.Thread(target=threaded_record, args=(audio_file_path, 5))
            thread.start()

            # Update countdown next to button
            for i in range(5, 0, -1):
                status_placeholder.markdown(f"**🎤 Recording... {i} seconds remaining**")
                time.sleep(1)

            thread.join()
            status_placeholder.empty()

            # Transcribe and respond
            question = transcribe_audio(audio_file_path)

            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.answer_question(
                            question,
                            st.session_state.current_video.video_id,
                            st.session_state.current_video.title,
                            st.session_state.current_video.language
                        )
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
        # Handle text input
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.answer_question(
                        prompt, 
                        st.session_state.current_video.video_id,
                        st.session_state.current_video.title,
                        st.session_state.current_video.language
                    )
                st.markdown(response)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("👈 Please load a YouTube video from the sidebar to start chatting!")
        
        # Instructions
        st.markdown("""
        ## How to use:
        1. **Enter a YouTube URL** in the sidebar
        2. **Click "Process Video"** to analyze the content
        3. **Ask questions** about the video content
        4. **Get AI-powered answers** based on the video transcript
        
        ## Features:
        - 🎯 **RAG System**: Uses vector search to find relevant content
        - 🧠 **Gemini 2.0 Flash**: Advanced AI for transcription and Q&A
        - 🌍 **Multi-language Support**: Works with 25+ languages
        - 📝 **Auto-transcription**: Works even without existing captions
        - 🔄 **Translation**: Can translate transcripts when needed
        - 💾 **Vector Storage**: Efficient content retrieval with Pinecone
        """)

if __name__ == "__main__":
    main()
