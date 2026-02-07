"""
Translation and Text-to-Speech Module
Provides translation and TTS capabilities for ASL recognition output.
"""

from typing import Optional
import logging
from gtts import gTTS
from googletrans import Translator
import io

logger = logging.getLogger(__name__)


class TranslationTTS:
    """
    Handles translation and text-to-speech conversion.
    Uses Google Translate API and Google Text-to-Speech.
    """
    
    def __init__(self):
        """Initialize the translation and TTS service."""
        self.translator = Translator()
        logger.info("Translation/TTS service initialized")
    
    def translate(self, text: str, target_language: str = 'es', source_language: str = 'en') -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (ISO 639-1)
            source_language: Source language code (default: 'en' for English)
            
        Returns:
            Translated text, or original text if translation fails
        """
        if not text or not text.strip():
            return ""
        
        # If target is English, no translation needed
        if target_language == 'en':
            return text
        
        try:
            translation = self.translator.translate(
                text,
                src=source_language,
                dest=target_language
            )
            
            if translation and translation.text:
                logger.debug(f"Translated '{text}' to '{translation.text}'")
                return translation.text
            else:
                logger.warning(f"Translation returned empty result for: {text}")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def text_to_speech(self, text: str, language: str = 'es') -> Optional[bytes]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            language: Language code for TTS (ISO 639-1)
            
        Returns:
            MP3 audio data as bytes, or None if conversion fails
        """
        if not text or not text.strip():
            return None
        
        try:
            # Create TTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            audio_data = audio_buffer.read()
            logger.debug(f"Generated TTS audio for: {text[:50]}...")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def translate_and_speak(self, text: str, target_language: str = 'es') -> tuple[str, Optional[bytes]]:
        """
        Translate text and generate speech for the translation.
        
        Args:
            text: Text to translate and speak
            target_language: Target language code
            
        Returns:
            Tuple of (translated_text, audio_data)
        """
        # Translate
        translated = self.translate(text, target_language=target_language)
        
        # Generate speech
        audio = self.text_to_speech(translated, language=target_language)
        
        return translated, audio
    
    def cleanup(self):
        """Cleanup resources (placeholder for future extensions)."""
        logger.info("Translation/TTS cleanup complete")


# Language code mappings (for reference)
LANGUAGE_CODES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'zh-CN': 'Chinese (Simplified)',
    'zh-TW': 'Chinese (Traditional)',
    'ko': 'Korean',
    'ar': 'Arabic',
    'ru': 'Russian',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'no': 'Norwegian'
}


def get_language_name(code: str) -> str:
    """
    Get language name from code.
    
    Args:
        code: ISO 639-1 language code
        
    Returns:
        Language name or the code if not found
    """
    return LANGUAGE_CODES.get(code, code)
