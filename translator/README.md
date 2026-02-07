# Real-time ASL Recognition Application

A Flask-based web application for real-time American Sign Language (ASL) recognition with dual-engine support for both sentence-level and alphabet-level recognition.

## Features

- **Dual Engine Architecture**: Runtime switching between sentence and alphabet recognition
- **Real-time Video Processing**: Live webcam feed with ASL detection overlay
- **Translation & TTS**: Automatic translation to multiple languages with text-to-speech
- **Prediction Smoothing**: Separate smoothing buffers for each engine to improve accuracy
- **Thread-Safe Operations**: Proper locking mechanisms for concurrent access
- **Dynamic Control**: Toggle detection, switch engines, and change languages on the fly

## Architecture Overview

### Engine Management

The application maintains two separate ASL recognition engines:

1. **Sentence Engine** (`RealtimeASLEngine`): Recognizes complete ASL sentences/phrases
2. **Alphabet Engine** (`RealtimeAlphabetEngine`): Recognizes individual ASL letters

Only one engine is active at a time, determined by `current_engine_type` ('sentence' or 'alphabet').

### Smoothing Buffers

Each engine has its own smoothing buffer to prevent prediction mixing when switching:
- `sentence_smoothing_buffer`: Stores predictions from the sentence engine
- `alphabet_smoothing_buffer`: Stores predictions from the alphabet engine

This ensures that switching engines doesn't carry over incorrect predictions.

### Thread Safety

The application uses multiple thread locks:
- `engine_lock`: Protects engine switching operations
- `translation_lock`: Protects shared output variables
- `frame_lock`: (Reserved for frame access if needed)

## API Endpoints

### GET /

Serves the main application HTML interface.

**Response**: HTML page

---

### GET /video_feed

Streams video frames with ASL detection overlay.

**Response**: MJPEG stream (multipart/x-mixed-replace)

**Features**:
- Real-time frame processing
- Status overlay (engine type, detection state)
- Automatic frame encoding and streaming

---

### GET /get_translation

Retrieves the latest ASL sentence and its translation.

**Response**:
```json
{
  "sentence": "Hello world",
  "translation": "Hola mundo",
  "audio": "base64_encoded_audio_data",
  "engine": "sentence"
}
```

**Fields**:
- `sentence`: Recognized ASL text
- `translation`: Translated text in target language
- `audio`: Base64-encoded MP3 audio of translation
- `engine`: Currently active engine type

---

### POST /toggle_detection

Toggles ASL detection on/off.

**Request**: No body required

**Response**:
```json
{
  "success": true,
  "detection_enabled": true
}
```

---

### POST /set_language

Sets the target language for translation.

**Request**:
```json
{
  "language": "es"
}
```

**Supported Language Codes**:
- `es`: Spanish
- `fr`: French
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- (Any ISO 639-1 two-letter code)

**Response**:
```json
{
  "success": true,
  "language": "es"
}
```

---

### POST /set_engine

Switches between sentence and alphabet recognition engines.

**Request**:
```json
{
  "engine": "alphabet"
}
```

**Valid Values**: `"sentence"` or `"alphabet"`

**Response**:
```json
{
  "success": true,
  "engine": "alphabet"
}
```

**Behavior**:
- Clears `latest_sentence` when switching
- Maintains separate smoothing buffers
- Does NOT clear smoothing buffer history (allows quick switching back)

**Error Response** (400):
```json
{
  "success": false,
  "error": "Invalid engine type. Must be 'sentence' or 'alphabet'"
}
```

---

### POST /clear_sentence

Clears the current sentence and translation.

**Request**: No body required

**Response**:
```json
{
  "success": true
}
```

**Use Case**: Useful in alphabet mode to start spelling a new word.

---

### GET /get_status

Retrieves current application status.

**Response**:
```json
{
  "detection_enabled": true,
  "current_engine": "sentence",
  "current_language": "es",
  "sentence_engine_available": true,
  "alphabet_engine_available": true,
  "webcam_available": true,
  "translator_available": true
}
```

## Configuration

### Smoothing Parameters

Located at the top of `app.py`:

```python
SMOOTHING_WINDOW_SIZE = 5  # Number of frames for smoothing
CONFIDENCE_THRESHOLD = 0.6  # Minimum prediction confidence
```

**Smoothing Window Size**: Larger values = more stable but slower to update  
**Confidence Threshold**: Higher values = fewer false positives but may miss some signs

### Model Paths

Engines expect the following files in the application directory:

- `sentence_model.h5`: Sentence recognition model
- `sentence_labels.txt`: Sentence class labels
- `alphabet_model.h5`: Alphabet recognition model
- `alphabet_labels.txt`: Alphabet class labels

## Usage Examples

### Basic Workflow

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   ```
   http://localhost:5000
   ```

3. **Default behavior**:
   - Sentence engine is active
   - Detection is enabled
   - Default language is Spanish (es)

### Switching to Alphabet Mode

```javascript
// JavaScript example
fetch('/set_engine', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({engine: 'alphabet'})
})
.then(response => response.json())
.then(data => console.log('Engine switched:', data.engine));
```

### Clearing Text in Alphabet Mode

```javascript
fetch('/clear_sentence', {
  method: 'POST'
})
.then(response => response.json())
.then(data => console.log('Sentence cleared'));
```

### Changing Translation Language

```javascript
fetch('/set_language', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({language: 'fr'})
})
.then(response => response.json())
.then(data => console.log('Language set to:', data.language));
```

### Polling for Updates

```javascript
// Poll every 500ms for new translations
setInterval(() => {
  fetch('/get_translation')
    .then(response => response.json())
    .then(data => {
      document.getElementById('sentence').textContent = data.sentence;
      document.getElementById('translation').textContent = data.translation;
      
      // Play audio if available
      if (data.audio) {
        const audio = new Audio('data:audio/mp3;base64,' + data.audio);
        audio.play();
      }
    });
}, 500);
```

## Engine-Specific Behavior

### Sentence Engine

- **Output**: Complete phrases (e.g., "Hello", "Thank you", "I love you")
- **Update Mode**: Replaces entire sentence with new prediction
- **Best For**: Natural conversation, common phrases

### Alphabet Engine

- **Output**: Individual letters (A-Z)
- **Update Mode**: Appends letters to build words
- **Best For**: Spelling names, uncommon words, fingerspelling
- **Note**: Use `/clear_sentence` to start a new word

## Dependencies

Required Python packages:
```
Flask>=2.3.0
opencv-python>=4.8.0
numpy>=1.24.0
tensorflow>=2.13.0
mediapipe>=0.10.0
gtts>=2.3.0
googletrans>=4.0.0
```

## Error Handling

The application includes comprehensive error handling:

- **Engine Initialization Failures**: Logged and app exits with code 1
- **Webcam Access Issues**: Logged and app exits with code 1
- **Translation Errors**: Logged but app continues without translation
- **Frame Processing Errors**: Logged and skipped, next frame processed
- **Invalid API Requests**: Returns 400/500 with error message

## Resource Cleanup

The application properly cleans up resources on shutdown:
- Releases webcam
- Cleans up TensorFlow/MediaPipe sessions
- Closes translation services

Cleanup is triggered by:
- Normal shutdown (Ctrl+C)
- Application errors
- `atexit` handler

## Performance Considerations

1. **Frame Rate**: Processing runs as fast as the webcam allows
2. **Engine Switching**: Instant, no reload required
3. **Memory**: Both engines loaded in memory simultaneously (~500MB-2GB depending on models)
4. **CPU/GPU**: Uses TensorFlow backend (GPU if available)

## Troubleshooting

### Video feed not showing
- Check webcam permissions
- Verify webcam is not in use by another application
- Check browser console for errors

### Predictions not updating
- Verify detection is enabled (`/toggle_detection`)
- Check confidence threshold isn't too high
- Ensure lighting is adequate for MediaPipe hand detection

### Engine switch not working
- Check that both model files exist
- Verify engine initialization in logs
- Ensure request has correct JSON format

### Translation not working
- Check internet connection (required for Google Translate)
- Verify translator initialized successfully in logs
- Check language code is valid ISO 639-1

## Development

### Running in Debug Mode

Change in `app.py`:
```python
app.run(debug=True, use_reloader=False)
```

**Note**: Keep `use_reloader=False` to prevent double initialization of engines.

### Adding New Engines

1. Implement engine class with `process_frame()` method
2. Add initialization in `initialize_engines()`
3. Add engine type to `set_engine()` validation
4. Update `get_active_engine()` logic
5. Add corresponding smoothing buffer

## License

[Your License Here]

## Support

For issues or questions, please [contact/file an issue].
