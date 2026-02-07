"""
Translation and Text-to-Speech module for ASL demo.
Handles Google Translate API and gTTS for speech output.
"""

import os
from gtts import gTTS
from googletrans import Translator
import io
import base64
from typing import Optional, Tuple
import tempfile


class TranslationTTSEngine:
    """
    Translation and Text-to-Speech engine.
    """
    
    def __init__(self, config: dict):
        """
        Initialize translation and TTS engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.translator = Translator()
        
        # Cache for audio files to avoid regenerating
        self.audio_cache = {}
        self.cache_max_size = 50
        
        # Statistics
        self.total_translations = 0
        self.total_tts_generations = 0
    
    def translate(self, text: str, target_lang: str = 'es') -> Tuple[str, str]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'es', 'fr')
        
        Returns:
            (translated_text, detected_source_lang)
        """
        if not text or not self.config['translation']['enabled']:
            return text, 'en'
        
        try:
            result = self.translator.translate(text, dest=target_lang)
            self.total_translations += 1
            return result.text, result.src
        except Exception as e:
            print(f"Translation error: {e}")
            return text, 'en'
    
    def text_to_speech(self, text: str, lang: str = 'en', 
                       return_base64: bool = True) -> Optional[str]:
        """
        Convert text to speech using gTTS.
        
        Args:
            text: Text to convert
            lang: Language code
            return_base64: If True, return base64 encoded audio
        
        Returns:
            Base64 encoded MP3 audio or file path
        """
        if not text or not self.config['tts']['enabled']:
            return None
        
        # Check cache
        cache_key = f"{text}_{lang}"
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        
        try:
            # Generate speech
            tts = gTTS(
                text=text,
                lang=lang,
                slow=self.config['tts']['slow']
            )
            
            if return_base64:
                # Save to bytes buffer
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                # Encode to base64
                audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
                
                # Cache result
                if len(self.audio_cache) < self.cache_max_size:
                    self.audio_cache[cache_key] = audio_base64
                
                self.total_tts_generations += 1
                return audio_base64
            else:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    self.total_tts_generations += 1
                    return fp.name
        
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    def process_sentence(self, sentence: str, target_lang: str = 'es'
                        ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Complete processing: translate and generate speech.
        
        Args:
            sentence: Input sentence
            target_lang: Target language for translation
        
        Returns:
            (translated_text, original_audio_base64, translated_audio_base64)
        """
        if not sentence:
            return "", None, None
        
        # Translate
        translated, src_lang = self.translate(sentence, target_lang)
        
        # Generate TTS for original (English)
        original_audio = self.text_to_speech(sentence, lang='en')
        
        # Generate TTS for translation
        translated_audio = self.text_to_speech(translated, lang=target_lang)
        
        return translated, original_audio, translated_audio
    
    def clear_cache(self):
        """Clear audio cache."""
        self.audio_cache.clear()
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            'total_translations': self.total_translations,
            'total_tts_generations': self.total_tts_generations,
            'cache_size': len(self.audio_cache)
        }


def test_translation_tts():
    """Test translation and TTS functionality."""
    print("Testing Translation & TTS Engine...")
    print("=" * 60)
    
    config = {
        'translation': {'enabled': True},
        'tts': {'enabled': True, 'slow': False}
    }
    
    engine = TranslationTTSEngine(config)
    
    # Test translation
    print("\n1. Testing translation...")
    text = "hello world"
    
    for lang, name in [('es', 'Spanish'), ('fr', 'French'), ('de', 'German')]:
        translated, src = engine.translate(text, lang)
        print(f"  {name}: {translated}")
    
    # Test TTS
    print("\n2. Testing TTS...")
    audio = engine.text_to_speech("hello world", lang='en')
    
    if audio:
        print(f"  ✓ Generated audio (base64 length: {len(audio)})")
    else:
        print("  ✗ TTS failed")
    
    # Test full pipeline
    print("\n3. Testing full pipeline...")
    translated, orig_audio, trans_audio = engine.process_sentence(
        "hello world", target_lang='es'
    )
    
    print(f"  Translated: {translated}")
    print(f"  Original audio: {'✓' if orig_audio else '✗'}")
    print(f"  Translated audio: {'✓' if trans_audio else '✗'}")
    
    # Statistics
    print("\n4. Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_translation_tts()
