import streamlit as st
import json
import random
import threading
import time
import logging
import traceback
import os
from tqdm import tqdm
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a rate limiter class to enforce request limits
class RateLimiter:
    def __init__(self, max_calls_per_minute=5):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait if we're at the rate limit"""
        current_time = time.time()
        with self.lock:
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls
                          if current_time - call_time < 60]
            
            # If we're at the limit, wait until we can make another call
            if len(self.calls) >= self.max_calls_per_minute:
                sleep_time = 60 - (current_time - self.calls[0]) + 0.1  # Add a small buffer
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            
            # Record this call
            self.calls.append(time.time())

class GeminiQuestionGenerator:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        # Initialize the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize model configuration
        self.model_name = model_name
        
        # Question history to prevent duplicates
        self.question_history = set()
        
        # Define response schema for structured output
        self.response_schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "question_image_url": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "image_url": {"type": "string"}
                                    },
                                    "required": ["text", "image_url"]
                                }
                            },
                            "correct_option_index": {"type": "integer"},
                            "explanation": {"type": "string"},
                            "topic": {"type": "string"}
                        },
                        "required": ["question_text", "options", "correct_option_index", "explanation", "topic"]
                    }
                }
            },
            "required": ["questions"]
        }
        
        self.generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": self.response_schema
        }
        
        self.safety_settings = {
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL': 'BLOCK_NONE',
            'DANGEROUS': 'BLOCK_NONE',
            'HATE_SPEECH': 'BLOCK_NONE',
            'SEXUALLY_EXPLICIT': 'BLOCK_NONE'
        }
        
        # Create rate limiter
        self.rate_limiter = RateLimiter(max_calls_per_minute=5)
    
    def generate_questions(self, biology_data, num_questions=10, selected_topics=None):
        """
        Generate MCQ questions using Gemini API based on the biology data
        
        Args:
            biology_data: JSON data containing biology chapter information
            num_questions: Number of questions to generate
            selected_topics: List of topics to include (None for all)
            
        Returns:
            List of generated questions with the response schema format
        """
        self.rate_limiter.wait_if_needed()
        
        # Initialize all_questions list at the beginning
        all_questions = []
        
        # Initialize question history from existing questions.json if it exists
        self._initialize_history()
        
        # Extract topics_subtopics from the biology data
        topics_subtopics = []
        for chapter in biology_data:
            topics_subtopics.extend(chapter.get('topics_subtopics', []))
        
        # Filter by selected topics if provided
        if selected_topics and topics_subtopics:
            filtered_data = [item for item in topics_subtopics 
                            if item.get('topic') in selected_topics]
        else:
            filtered_data = topics_subtopics
        
        if not filtered_data:
            logger.error("No filtered data available to generate questions")
            return []
            
        # Prepare image data to send to Gemini
        image_data = []
        for item in filtered_data:
            if item.get('image_url') and item.get('is_valid', False):
                image_data.append({
                    'image_url': item.get('image_url'),
                    'description': item.get('image_description', ''),
                    'title': item.get('image_title', ''),
                    'context': item.get('context', ''),
                    'topic': item.get('topic', ''),
                    'subtopic': item.get('subtopic', '')
                })
        
        if not image_data:
            logger.error("No valid image data found to generate questions")
            return []

        # Shuffle all image data to ensure randomness
        random.shuffle(image_data)
        logger.info(f"Found {len(image_data)} valid images for question generation (shuffled)")
        
        # For larger question counts, process in chunks to avoid API limitations
        img_question_count = 0
        img_option_count = 0
        chunk_size = 10  # Gemini works best with generating 10 questions at a time
        remaining = num_questions
        
        while remaining > 0:
            chunk_questions = min(chunk_size, remaining)
            
            # Calculate how many of each type we should have
            curr_ratio = 0 if not all_questions else img_option_count / len(all_questions)
            target_img_option_ratio = 0.5  # We want at least 50% image option questions
            
            # Shuffle image data again for each batch to maximize variety
            random.shuffle(image_data)
            batch_images = image_data[:50]  # Take the first 50 images after shuffling
            
            # Prepare the prompt for Gemini with the current chunk
            prompt = f"""
            Create exactly {chunk_questions} multiple-choice biology questions based on the following image data.
            
            IMPORTANT REQUIREMENTS:
            1. EACH question MUST have EXACTLY 4 options (no more, no less)
            2. For all questions, provide options labeled A, B, C, and D
            3. Each option must be distinct and meaningful
            4. Always have a sweet balance of Image based questions and image option based questions.
            Create a mix of question types as follows:
            - AT LEAST 50% of questions MUST be "Text Question + Image Options" type
            - The remaining questions should be "Image in Question + Text Options" type
            
            Types of questions to create:
            a) "Image in Question + Text Options": Use this when the image shows a specific structure, process, or organism that students need to identify or analyze. The image should be in the question, and the options should be text-only choices.
            
            b) "Text Question + Image Options": Use this when asking students to select a visual example that matches a description. The question should be text-only, and the options should be images. GENERATE AT LEAST {max(3, int(chunk_questions*0.5))} QUESTIONS OF THIS TYPE.
            
            For "Text Question + Image Options" questions:
            - Make the question text-only (NO image in the question)
            - Make ALL 4 options have valid image URLs 
            - All 4 image options should be of the same category (e.g., all cell types, all plant species, etc.)
            - Examples: "Which of the following images shows a prokaryotic cell?", "Select the image that represents an example of commensalism."
            
            For "Image in Question + Text Options" questions:
            - Include ONE image in the question field
            - Make all 4 options TEXT-ONLY (empty image_url field)
            - Reference the image explicitly: "In the image above..."
            
            IMPORTANT: A question must NEVER have both an image in the question AND images in the options.
            
            For EACH image option question:
            - Make sure all option images have valid URLs
            - Ensure options are visually distinguishable
            - Set the text field to "" for all image options
            
            Use the image_url field for showing images.
            Use the topic field to categorize questions.
            
            CRITICAL: EVERY question MUST have EXACTLY 4 options - questions with fewer options will be rejected. 
            
            
            Here is the image data: {json.dumps(batch_images)}
            """
            
            try:
                # Create the Gemini model
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Generate response
                logger.info(f"Generating batch of {chunk_questions} questions")
                response = model.generate_content(prompt)
                
                # Parse the response
                if hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        # Extract JSON from the response
                        response_text = content.parts[0].text
                        logger.info(f"Received response of length: {len(response_text)}")
                        
                        # Sometimes Gemini wraps JSON in markdown code blocks, so we need to extract it
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].strip()
                        
                        # Parse the JSON
                        try:
                            questions_data = json.loads(response_text)
                            # Use only questions that follow our schema
                            if 'questions' in questions_data:
                                # Validate and process questions
                                validated_questions = self._validate_questions(questions_data['questions'])
                                
                                # Count question types
                                for q in validated_questions:
                                    # Check if this is an image-option question (no question image, but has image options)
                                    has_question_image = bool(q.get('question_image_url'))
                                    has_image_options = any(opt.get('image_url') for opt in q.get('options', []))
                                    
                                    if not has_question_image and has_image_options:
                                        img_option_count += 1
                                    elif has_question_image and not has_image_options:
                                        img_question_count += 1
                                
                                # Filter out any duplicate questions
                                non_duplicate_questions = self._filter_duplicates(validated_questions)
                                
                                logger.info(f"Successfully validated {len(non_duplicate_questions)} unique questions from this batch")
                                logger.info(f"Image option questions: {img_option_count}, Image question questions: {img_question_count}")
                                
                                all_questions.extend(non_duplicate_questions)
                                
                                # Update history with new questions
                                for q in non_duplicate_questions:
                                    self._add_to_history(q)
                            else:
                                logger.warning("No 'questions' key found in response JSON")
                                
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON from Gemini response: {response_text[:200]}")
                
                # Fall back to loading from questions.json if still not enough questions
                if not all_questions and os.path.exists('questions.json'):
                    logger.info("Attempting to load questions from questions.json as fallback")
                    try:
                        with open('questions.json', 'r') as f:
                            fallback_data = json.load(f)
                            if isinstance(fallback_data, list):
                                all_questions = fallback_data[:num_questions]
                                logger.info(f"Loaded {len(all_questions)} questions from questions.json")
                    except Exception as e:
                        logger.error(f"Error loading questions.json: {str(e)}")
                
                # If we got some questions, save the progress
                if all_questions and len(all_questions) > 0:
                    remaining = num_questions - len(all_questions)
                    
                    # Save the current progress to questions.json
                    try:
                        with open('questions.json', 'w') as f:
                            json.dump(all_questions, f, indent=2)
                            logger.info(f"Saved {len(all_questions)} questions to questions.json")
                    except Exception as e:
                        logger.error(f"Error saving questions.json: {str(e)}")
                    
                    # Also save to the question database
                    self.save_to_database(all_questions)
                else:
                    # If no questions were generated, break the loop to avoid infinite loops
                    logger.error("Failed to generate any questions in this batch")
                    break
                
            except Exception as e:
                logger.error(f"Error generating questions with Gemini: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        return all_questions

    def save_to_database(self, questions, file_path="question_database.json"):
        """
        Save questions to the question database file, avoiding duplicates
        
        Args:
            questions: List of questions to save
            file_path: Path to the question database file
            
        Returns:
            Number of new questions added to the database
        """
        # Load existing questions from the database
        existing_questions = []
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    existing_questions = json.load(f)
                    logger.info(f"Loaded {len(existing_questions)} existing questions from database")
        except Exception as e:
            logger.error(f"Error loading existing questions from database: {str(e)}")
            # Continue with an empty list if there was an error
            existing_questions = []
        
        # Create a set of question fingerprints to avoid duplicates
        existing_fingerprints = set()
        for q in existing_questions:
            question_text = q.get('question_text', '').strip().lower()
            question_image = q.get('question_image_url', '')
            fingerprint = f"{question_text[:100]}|{question_image[:50]}"
            existing_fingerprints.add(fingerprint)
        
        # Add only new questions that don't already exist in the database
        new_questions_count = 0
        for q in questions:
            question_text = q.get('question_text', '').strip().lower()
            question_image = q.get('question_image_url', '')
            fingerprint = f"{question_text[:100]}|{question_image[:50]}"
            
            if fingerprint not in existing_fingerprints:
                existing_questions.append(q)
                existing_fingerprints.add(fingerprint)
                new_questions_count += 1
        
        # Save the updated database
        if new_questions_count > 0:
            try:
                with open(file_path, 'w') as f:
                    json.dump(existing_questions, f, indent=2)
                    logger.info(f"Added {new_questions_count} new questions to database")
            except Exception as e:
                logger.error(f"Error saving question database: {str(e)}")
        else:
            logger.info("No new questions to add to database")
        
        return new_questions_count

    def _validate_questions(self, questions):
        """Validate and clean up questions from Gemini"""
        validated = []
        
        # Define allowed domains for image URLs
        allowed_domains = [
            "ncert-holistic-data-extraction.s3.amazonaws.com",
            "upload.wikimedia.org"  # Add other trusted domains if needed
        ]
        
        for q in questions:
            # Ensure required fields are present
            if not all(key in q for key in ['question_text', 'options', 'correct_option_index']):
                continue
                
            # Ensure options list is valid and has exactly 4 options
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                # Skip questions with fewer than 4 options
                logger.warning(f"Skipping question with {len(q['options'])} options instead of 4")
                continue
                
            # Ensure correct_option_index is valid
            if not isinstance(q['correct_option_index'], int) or q['correct_option_index'] >= len(q['options']):
                continue
                
            # Ensure all options have required fields
            valid_options = True
            for opt in q['options']:
                if not isinstance(opt, dict) or not all(key in opt for key in ['text', 'image_url']):
                    valid_options = False
                    break
            
            if not valid_options:
                continue
                
            # Validate image URLs to ensure they're from trusted domains
            question_image_url = q.get('question_image_url', '')
            if question_image_url and not any(domain in question_image_url for domain in allowed_domains):
                # Skip questions with untrusted image URLs
                logger.warning(f"Skipping question with untrusted image URL: {question_image_url[:50]}...")
                continue
                
            # Validate option image URLs
            for opt in q['options']:
                image_url = opt.get('image_url', '')
                if image_url and not any(domain in image_url for domain in allowed_domains):
                    valid_options = False
                    logger.warning(f"Found option with untrusted image URL: {image_url[:50]}...")
                    break
            
            if not valid_options:
                continue
                
            # Add explanation if missing
            if 'explanation' not in q:
                q['explanation'] = "No explanation provided."
                
            # Add topic if missing
            if 'topic' not in q:
                q['topic'] = "General Biology"
                
            validated.append(q)
        
        return validated
    
    def _initialize_history(self):
        """Initialize question history from questions.json if it exists"""
        if os.path.exists('questions.json'):
            try:
                with open('questions.json', 'r') as f:
                    questions = json.load(f)
                    for q in questions:
                        self._add_to_history(q)
                    logger.info(f"Initialized history with {len(self.question_history)} questions from questions.json")
            except Exception as e:
                logger.error(f"Error initializing question history: {str(e)}")
    
    def _add_to_history(self, question):
        """Add a question to history to prevent duplicates"""
        # Create a unique fingerprint for the question using text and topic
        question_text = question.get('question_text', '').strip().lower()
        question_image = question.get('question_image_url', '')
        
        # Only use first 100 chars to avoid minor differences causing duplicates
        fingerprint = f"{question_text[:100]}|{question_image[:50]}"
        self.question_history.add(fingerprint)
    
    def _filter_duplicates(self, questions):
        """Filter out questions that match our history"""
        unique_questions = []
        for q in questions:
            question_text = q.get('question_text', '').strip().lower()
            question_image = q.get('question_image_url', '')
            fingerprint = f"{question_text[:100]}|{question_image[:50]}"
            
            if fingerprint not in self.question_history:
                unique_questions.append(q)
        
        return unique_questions
    
    def upload_to_gemini(self, path, mime_type=None):
        """
        Uploads the given file to Gemini.
        
        Args:
            path: Path to the file to upload
            mime_type: MIME type of the file (optional)
            
        Returns:
            Uploaded file object
        """
        try:
            file = genai.upload_file(path, mime_type=mime_type)
            logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file
        except Exception as e:
            logger.error(f"Error uploading file to Gemini: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def wait_for_files_active(self, files):
        """
        Polls until all uploaded files are processed and active.
        
        Args:
            files: List of file objects returned by upload_to_gemini
            
        Returns:
            None
        """
        logger.info("Waiting for file processing...")
        for file in tqdm(files, desc="Processing files"):
            pbar = tqdm(total=100, desc=f"File: {file.display_name}", leave=False)
            while file.state.name == "PROCESSING":
                pbar.update(10)  # Update progress bar
                time.sleep(10)
                file = genai.get_file(file.name)
            pbar.close()
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        logger.info("All files active and ready for processing")

def load_biology_data(file_path):
    """Load biology chapter data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return []

def load_all_biology_data():
    """Load biology chapter data from all JSON files in the current folder"""
    all_data = []
    json_files = [f for f in os.listdir('.') if f.lower().endswith('.json') and 'biology' in f.lower()]
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    logger.info(f"Loaded data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
    
    return all_data

def get_available_topics(biology_data):
    """Extract unique topics from the biology chapter data"""
    topics = set()
    for chapter in biology_data:
        for item in chapter.get('topics_subtopics', []):
            if 'topic' in item:
                topics.add(item['topic'])
    return sorted(list(topics))

def load_question_database(file_path="question_database.json", num_questions=None, selected_topics=None):
    """
    Load questions from the question database JSON file
    
    Args:
        file_path: Path to the question database JSON file
        num_questions: Number of questions to return (None for all)
        selected_topics: List of topics to filter by (None for all)
        
    Returns:
        List of questions from the database
    """
    try:
        with open(file_path, 'r') as f:
            questions = json.load(f)
            
        logger.info(f"Loaded {len(questions)} questions from database")
        
        # Filter by selected topics if provided
        if selected_topics:
            questions = [q for q in questions if q.get('topic') in selected_topics]
            logger.info(f"Filtered to {len(questions)} questions by topic")
            random.shuffle(questions)  # Shuffle the filtered questions
        # Select random subset if num_questions specified
        if num_questions and num_questions < len(questions):
            return random.sample(questions, num_questions)
        
        return questions
    except Exception as e:
        logger.error(f"Error loading question database: {str(e)}")
        return []

def main():
    st.set_page_config(page_title="Biology MCQ Quiz Generator", layout="wide")
    
    # Load API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key.")
        st.stop()
    
    # Check if we're in the results page
    if 'quiz_completed' in st.session_state and st.session_state.quiz_completed:
        show_results_page()
        return
    
    # Main title
    st.title("Biology Image-based MCQ Quiz Generator")
    
    # Load data from all JSON files
    biology_data = load_all_biology_data()
    
    if biology_data:
        # Get available topics
        topics = get_available_topics(biology_data)
        
        # Configuration section in main area
        st.header("Quiz Configuration")
        
        # Add option to choose question source
        question_source = st.radio(
            "Question Source",
            options=["Use Existing Questions", "Generate New Questions"],
            index=0  # Default to using existing questions
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Topic selection
            selected_topics = st.multiselect(
                "Select topics (leave empty for all)",
                options=topics,
                default=[]
            )
        
        with col2:
            # Number of questions
            num_questions = st.slider(
                "Number of questions",
                min_value=1,
                max_value=50,
                value=10
            )
        
        # Generate button
        if st.button("Generate Questions", type="primary"):
            if question_source == "Generate New Questions" and not api_key:
                st.error("Gemini API Key not available. Please check your .env file.")
            else:
                if not selected_topics:
                    selected_topics = topics  # Use all topics if none selected
                
                with st.spinner("Preparing questions..."):
                    try:
                        if question_source == "Generate New Questions":
                            generator = GeminiQuestionGenerator(api_key=api_key)
                            
                            # Use Gemini for question generation
                            questions = generator.generate_questions(
                                biology_data,
                                num_questions=num_questions,
                                selected_topics=selected_topics
                            )
                            
                            # Display a message about questions being saved to database
                            if questions:
                                st.success(f"Generated {len(questions)} questions! These have been saved to the database.")
                        else:
                            # Load questions from database
                            questions = load_question_database(
                                num_questions=num_questions,
                                selected_topics=selected_topics
                            )
                        
                        if not questions:
                            st.warning("No questions could be loaded or generated. Please try again with different settings.")
                        else:
                            # Store questions in session state
                            st.session_state.questions = questions
                            st.session_state.current_question = 0
                            st.session_state.answers = {}
                            st.session_state.quiz_started = True
                            
                            st.success(f"Loaded {len(questions)} questions!")
                    except Exception as e:
                        st.error(f"Error loading questions: {str(e)}")
                        st.exception(e)
        
        st.markdown("---")
    
    # Main content area for the quiz
    if 'quiz_started' in st.session_state and st.session_state.quiz_started and not ('quiz_completed' in st.session_state and st.session_state.quiz_completed):
        questions = st.session_state.questions
        current_idx = st.session_state.current_question
        
        if not questions:
            st.warning("No questions were generated. Try different topics or settings.")
            return
        
        # Create a two-column layout: main content and side panel
        main_col, side_col = st.columns([3, 1])
        
        with main_col:
            # Display progress
            st.progress((current_idx) / len(questions))
            st.write(f"Question {current_idx + 1} of {len(questions)}")
            
            if current_idx < len(questions):
                question = questions[current_idx]
                
                # Display the question
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(question["question_text"])
                    
                    # Display question image if available
                    if question.get("question_image_url"):
                        try:
                            st.image(question.get("question_image_url"), width=400)
                        except Exception as e:
                            st.error(f"Error displaying question image: {str(e)}")
                            st.warning("Image could not be displayed. URL may be invalid.")
                
                with col2:
                    st.write(f"Topic: {question.get('topic', 'General')}")
                
                # Determine type of options (text, image, or both)
                has_image_options = any(opt.get('image_url') for opt in question["options"])
                
                # Display options
                if has_image_options:
                    # If we have image options, use a 2-column layout
                    option_cols = st.columns(2)
                    for i, option in enumerate(question["options"]):
                        col_idx = i % 2
                        with option_cols[col_idx]:
                            option_container = st.container()
                            with option_container:
                                if option.get("image_url"):
                                    try:
                                        st.image(option["image_url"], width=200)
                                    except Exception as e:
                                        st.warning("⚠️ Option image could not be displayed")
                                
                                # For image-based options, only show the letter (A,B,C,D) without text
                                if st.button(f"{chr(65+i)}", key=f"option_{i}"):
                                    st.session_state.answers[current_idx] = i
                                    
                else:
                    # For text-only options, use a single column with radio buttons
                    options_text = [opt.get("text", f"Option {chr(65+i)}") for i, opt in enumerate(question["options"])]
                    selected = st.radio(
                        "Select your answer:",
                        options=options_text,
                        format_func=lambda i, idx=0: f"{chr(65+options_text.index(i))}. {i}",
                        key=f"radio_{current_idx}",
                        index=None
                    )
                    
                    if selected is not None:
                        selected_idx = options_text.index(selected)
                        st.session_state.answers[current_idx] = selected_idx
                
                # Next button to submit answer or mark as skipped
                st.markdown("---")
                next_col1, next_col2 = st.columns([3, 1])
                with next_col2:
                    if st.button("Next", type="primary"):
                        # If no answer is selected, question will be marked as skipped
                        # If an answer was selected, it's already stored in session_state.answers
                        if current_idx < len(questions) - 1:
                            st.session_state.current_question += 1
                            st.rerun()
                        else:
                            st.session_state.quiz_completed = True
                            st.rerun()
                
                # Navigation buttons
                nav_cols = st.columns(2)
                with nav_cols[0]:
                    if current_idx > 0:
                        if st.button("Previous Question"):
                            st.session_state.current_question -= 1
                            st.rerun()
                
                with nav_cols[1]:
                    if st.button("Submit Quiz"):
                        st.session_state.quiz_completed = True
                        st.rerun()
        
        # Side panel to show all question numbers and their status
        with side_col:
            st.markdown("### Questions")
            st.markdown("---")
            
            # Define colors for different question states
            answered_color = "#28a745"  # Green
            current_color = "#007bff"   # Blue
            skipped_color = "#dc3545"   # Red
            unattempted_color = "#6c757d" # Gray
            
            # Create a grid layout for question numbers
            num_columns = 3
            rows = [questions[i:i + num_columns] for i in range(0, len(questions), num_columns)]
            
            for row_idx, row in enumerate(rows):
                cols = st.columns(num_columns)
                for i, _ in enumerate(row):
                    q_idx = row_idx * num_columns + i
                    q_num = q_idx + 1
                    
                    # Determine the status and color of the question
                    if q_idx == current_idx:
                        color = current_color
                        status = "Current"
                    elif q_idx in st.session_state.answers:
                        color = answered_color
                        status = "Answered"
                    else:
                        color = unattempted_color
                        status = "Unattempted"
                    
                    # Create clickable button for each question number
                    with cols[i]:
                        if st.button(
                            f"{q_num}", 
                            key=f"nav_{q_idx}",
                            help=status,
                            use_container_width=True,
                            type="primary" if q_idx == current_idx else "secondary"
                        ):
                            st.session_state.current_question = q_idx
                            st.rerun()
            
            st.markdown("---")
            
            # Legend
            st.markdown("### Legend")
            st.markdown(f"<span style='color:{current_color}'>■</span> Current", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{answered_color}'>■</span> Answered", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{unattempted_color}'>■</span> Unattempted", unsafe_allow_html=True)

    elif not ('quiz_started' in st.session_state and st.session_state.quiz_started):
        st.write("""
        ## Biology MCQ Quiz Generator
        
        This application generates multiple-choice questions based on biology chapter data, including images and descriptions.
        
        ### Types of Questions:
        1. **Image in Question**: Questions that show an image and ask you to identify what it represents
        2. **Image Options**: Questions that ask you to select the correct image from multiple options
        3. **Both**: Questions that show an image and ask you to match it with another image
        
        ### How to use:
        1. Ensure your Gemini API Key is set in the .env file
        2. Select specific topics or use all available topics
        3. Choose the number of questions (default: 10)
        4. Click "Generate Questions" to create your MCQ exam
        5. Answer the questions and see your results!
        
        The application uses the Biology_chapter_F2.json file by default, which contains information about biological classification.
        """)
        
        # Display sample of the biology data
        st.subheader("Sample of Biology Chapter Data")
        try:
            biology_data = load_all_biology_data()
            if biology_data:
                sample_topic = biology_data[0]['topics_subtopics'][0]
                st.write(f"**Topic:** {sample_topic.get('topic', 'N/A')}")
                st.write(f"**Subtopic:** {sample_topic.get('subtopic', 'N/A')}")
                st.write("**Sample Context:**")
                st.write(sample_topic.get('context', 'N/A')[:200] + "...")
                
                if sample_topic.get('image_url'):
                    st.write("**Sample Image:**")
                    st.image(sample_topic.get('image_url'), width=300)
                    st.caption(sample_topic.get('image_title', ''))
        except Exception as e:
            st.error(f"Error displaying sample data: {str(e)}")

def show_results_page():
    """Display the quiz results on a separate page"""
    st.title("Quiz Results")
    
    questions = st.session_state.questions
    
    correct_count = 0
    attempted_count = 0
    
    for i, question in enumerate(questions):
        user_answer = st.session_state.answers.get(i)
        if user_answer is not None:
            attempted_count += 1
            correct_answer = question["correct_option_index"]
            if user_answer == correct_answer:
                correct_count += 1
    
    # Display score
    score_percentage = (correct_count / len(questions)) * 100 if questions else 0
    attempted_percentage = (attempted_count / len(questions)) * 100 if questions else 0
    
    st.header("Your Score Summary")
    score_col, attempted_col = st.columns(2)
    with score_col:
        st.metric("Your Score", f"{correct_count}/{len(questions)}", f"{score_percentage:.1f}%")
    with attempted_col:
        st.metric("Questions Attempted", f"{attempted_count}/{len(questions)}", f"{attempted_percentage:.1f}%")
    
    # Performance assessment
    if score_percentage >= 80:
        st.success("Excellent work! You have a strong understanding of the material.")
    elif score_percentage >= 60:
        st.info("Good job! You've demonstrated solid knowledge.")
    else:
        st.warning("You might want to review this material again.")
    
    # Detailed results
    st.markdown("---")
    st.subheader("Detailed Results")
    
    for i, question in enumerate(questions):
        user_answer = st.session_state.answers.get(i)
        correct_answer = question["correct_option_index"]
        
        # Set the status indicator
        if user_answer is None:
            status = "⚠️ Skipped"
        elif user_answer == correct_answer:
            status = "✅ Correct"
        else:
            status = "❌ Incorrect"
        
        with st.expander(f"Question {i+1}: {status} - {question['question_text'][:50]}..."):
            # Display question
            st.write(question["question_text"])
            
            if question.get("question_image_url"):
                try:
                    st.image(question["question_image_url"], width=300)
                except Exception as e:
                    st.warning("Image could not be displayed")
            
            # Display options
            st.write("**Options:**")
            
            # Check if this question has image-based options (all options have non-empty image_url)
            has_image_options = all(opt.get("image_url", "") != "" for opt in question["options"])
            
            for j, option in enumerate(question["options"]):
                prefix = ""
                if user_answer is not None and j == user_answer and j == correct_answer:
                    prefix = "✅ "
                elif user_answer is not None and j == user_answer:
                    prefix = "❌ "
                elif j == correct_answer:
                    prefix = "✓ "
                
                # Show option letter
                option_letter = f"{chr(65+j)}. "
                
                # For image-based options, only show the image and indicator
                if has_image_options:
                    st.write(f"{prefix}{option_letter}")
                    if option.get("image_url"):
                        try:
                            st.image(option.get("image_url"), width=150)
                        except Exception as e:
                            st.warning("Option image could not be displayed")
                else:
                    # For text options or mixed options, show both text and image if available
                    st.write(f"{prefix}{option_letter}{option.get('text', '')}")
                    if option.get("image_url"):
                        try:
                            st.image(option.get("image_url"), width=150)
                        except Exception as e:
                            st.warning("Option image could not be displayed")
            
            # Display explanation
            st.write("**Explanation:**")
            st.write(question.get("explanation", "No explanation provided."))
    
    # Restart button
    if st.button("Start New Quiz", type="primary"):
        for key in ['questions', 'current_question', 'answers', 'quiz_started', 'quiz_completed']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()



