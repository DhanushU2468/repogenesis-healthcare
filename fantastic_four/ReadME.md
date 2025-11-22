# AI Prescription Reader

Smart medication management system that reads prescription images and provides intelligent reminders.

## Features

- 📷 **Scan Prescriptions** - Upload prescription images and automatically extract medicine details
- 📅 **Smart Scheduling** - Generates complete medication schedule with timings
- 🔔 **Reminders** - Customizable notifications with sound and voice alerts
- 👨‍👩‍👧 **Family Dashboard** - Track medication adherence with statistics
- ✅ **Progress Tracking** - Mark medicines as taken/not taken

## Installation

### 1. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**  
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

```bash
pip install flask flask-cors pillow pytesseract werkzeug
```

### 3. Run the Application

```bash
python app.py
```

Open browser and go to: `http://localhost:5000`

## Usage

### Upload Prescription
1. Click "Upload" tab
2. Choose prescription image (PNG/JPG/JPEG)
3. Enter patient name and phone number
4. Click "Process Prescription"

### View Schedule
1. Go to "Schedule" tab
2. See all medicines with timings
3. Mark medicines as taken by clicking checkbox
4. Set custom reminders for specific medicines

### Enable Reminders
1. Click "Start Reminders" button
2. Customize reminder times (optional):
   - First reminder (default: 15 min before)
   - Second reminder (default: 5 min before)
   - Voice alert (default: 1 min before)
   - Overdue alert (default: 15 min after)
3. Toggle sound/voice as needed

### Family Dashboard
1. Go to "Family" tab
2. View adherence statistics
3. Check complete medication history

## Project Structure

```
fantastic_four/
├── app.py              # Backend server
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Prescription images (auto-created)
└── data/              # App data (auto-created)
```

## How It Works

1. **Image Upload** - User uploads prescription photo
2. **OCR Processing** - Tesseract extracts text from image
3. **Data Parsing** - Extracts medicine names, dosages, frequencies, durations
4. **Schedule Generation** - Creates complete medication timeline
5. **Smart Reminders** - Sends notifications at configured times

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and process prescription |
| GET | `/prescriptions` | List all prescriptions |
| GET | `/schedule/<id>` | Get medication schedule |
| GET | `/family_dashboard/<id>` | Get adherence statistics |
| POST | `/mark_taken` | Mark medicine as taken |

## Configuration

### Change Default Reminder Times
Edit values in the Reminder Controls section in the UI.

### Customize Image Processing
Modify `preprocess_image()` function in `app.py`:
- Contrast level (default: 2)
- Threshold value (default: 128)
- Image resize ratio (default: 1000px width)

## Troubleshooting

**OCR not extracting text?**
- Ensure Tesseract is installed and in PATH
- Use clear, well-lit prescription images
- Check that prescription has numbered medicine list (1., 2., 3.)

**Notifications not working?**
- Allow browser notification permissions
- Click "Start Reminders" button
- Check that prescription is uploaded

**Voice alerts not speaking?**
- Enable voice toggle in UI
- Check browser supports Web Speech API
- Try refreshing the page

## Browser Support

✅ Chrome/Edge | ✅ Firefox | ✅ Safari | ⚠️ Mobile (limited notifications)

## Security Notes

⚠️ **For Production:**
- Change `app.secret_key` in app.py
- Add user authentication
- Use HTTPS
- Add file upload validation
- Implement rate limiting

## Tech Stack & Dependencies

### Backend (Python)
- **Flask** `v2.0+` - Web framework (BSD-3-Clause License)
- **Flask-CORS** - Cross-Origin Resource Sharing (MIT License)
- **Pillow (PIL)** - Image processing library (HPND License)
- **pytesseract** - Python wrapper for Tesseract OCR (Apache 2.0 License)
- **Werkzeug** - WSGI utilities, bundled with Flask (BSD-3-Clause License)

### Frontend
- **Vanilla JavaScript** - No external frameworks
- **HTML5 & CSS3** - Modern web standards
- **Web Speech API** - Browser's built-in text-to-speech
- **Notification API** - Browser's notification system
- **Web Audio API** - For notification sounds

### External Dependencies (Open Source)
- **Tesseract OCR** - Open source OCR engine by Google (Apache 2.0 License)
  - Project: https://github.com/tesseract-ocr/tesseract
  - Required for text extraction from images

### Built-in Python Libraries (No Installation Required)
- `os` - File and directory operations
- `datetime` - Date and time handling
- `json` - JSON data handling
- `base64` - Base64 encoding/decoding
- `re` - Regular expressions for text parsing
- `session` - Session management (from Flask)

### Storage
- In-memory dictionaries (no external database)

## License

Free to use for personal and educational purposes.

---

**Made for better medication adherence** 💊