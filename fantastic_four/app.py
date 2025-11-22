from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import json
import base64
from werkzeug.utils import secure_filename
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path):
    """Enhanced image preprocessing for better OCR"""
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too small (improves OCR accuracy)
    width, height = img.size
    if width < 1000:
        ratio = 1000 / width
        img = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    # Apply threshold
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    return img

def extract_prescription_data(image_path):
    """
    Extract prescription data from image using advanced OCR.
    """
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Extract text with better config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        if not text.strip():
            return create_prescription_data(error='No text could be extracted from the image')
        
        print("\n" + "="*80)
        print("EXTRACTED TEXT FROM PRESCRIPTION")
        print("="*80)
        print(text)
        print("="*80 + "\n")
        
        # Parse prescription elements
        patient_info = extract_patient_info(text)
        medicines = extract_medicines_structured(text)
        
        prescription = {
            'medicines': medicines,
            'doctor_name': extract_doctor_name(text),
            'hospital': extract_hospital_name(text),
            'date': extract_date(text) or datetime.now().strftime('%Y-%m-%d'),
            'patient_name': patient_info.get('name', ''),
            'patient_age': patient_info.get('age', ''),
            'diagnosis': extract_diagnosis(text),
            'advice': extract_advice(text),
            'raw_text': text  # For debugging
        }
        
        return prescription
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_prescription_data(error=f'Error processing image: {str(e)}')

def create_prescription_data(error=None, **kwargs):
    """Helper function to create a standardized prescription data structure"""
    data = {
        'medicines': [],
        'doctor_name': '',
        'hospital': '',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'patient_name': '',
        'patient_age': '',
        'diagnosis': '',
        'advice': []
    }
    if error:
        data['error'] = error
    data.update(kwargs)
    return data

def extract_patient_info(text):
    """Extract patient name and age from text"""
    name_match = re.search(r'(?i)(?:patient|name)\s*[:\-]?\s*([A-Za-z\s.]+?)(?:\n|age|$)', text)
    age_match = re.search(r'(?i)age\s*[:\-]?\s*(\d+)', text)
    
    return {
        'name': name_match.group(1).strip() if name_match else '',
        'age': age_match.group(1) if age_match else ''
    }

def extract_medicines_structured(text):
    """
    NEW APPROACH: Extract medicines with complete context using block parsing.
    This groups all lines related to each medicine together.
    """
    medicines = []
    
    # Exclusion words
    exclude_words = {
        'dr', 'doctor', 'patient', 'name', 'age', 'date', 'prescription', 
        'diagnosis', 'advice', 'signature', 'hospital', 'clinic', 'medical',
        'tel', 'phone', 'address', 'email', 'reg', 'registration', 'no', 'number',
        'mbbs', 'md', 'ms', 'dnb', 'mrcp', 'frcs', 'speciality', 'consultant',
        'follow', 'next', 'visit', 'review', 'tests', 'investigations', 'rxs',
        'dear', 'sir', 'madam', 'regards', 'sincerely', 'yours', 'faithfully'
    }
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for numbered medicine items (1. Medicine, 2. Medicine)
        match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line)
        
        if match:
            med_number = match.group(1)
            rest_of_first_line = match.group(2).strip()
            
            # Extract medicine name from first part
            med_name_match = re.match(r'^([A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-z]+)?)', rest_of_first_line)
            
            if med_name_match:
                med_name = med_name_match.group(1).strip()
                
                # Check if it's a valid medicine name
                if len(med_name) >= 3 and med_name.lower() not in exclude_words:
                    print(f"\n{'='*60}")
                    print(f"Found medicine #{med_number}: {med_name}")
                    
                    # Collect all related lines (until next number or empty lines)
                    medicine_block = [rest_of_first_line]
                    j = i + 1
                    
                    # Gather next 6-8 lines or until we hit another numbered item
                    while j < len(lines) and j < i + 8:
                        next_line = lines[j].strip()
                        
                        # Stop if we hit another numbered item
                        if re.match(r'^\d+[\.\)]', next_line):
                            break
                        
                        # Stop if we hit doctor's signature or empty section
                        if not next_line or 'signature' in next_line.lower():
                            break
                            
                        medicine_block.append(next_line)
                        j += 1
                    
                    # Combine all lines into one text block for this medicine
                    full_medicine_text = ' '.join(medicine_block)
                    print(f"Full medicine block: {full_medicine_text}")
                    
                    # Extract details from the complete block
                    dosage = extract_dosage_from_text(full_medicine_text)
                    frequency = extract_frequency_from_text(full_medicine_text)
                    duration = extract_duration_from_block(full_medicine_text)
                    instructions = extract_instructions_from_text(full_medicine_text)
                    
                    print(f"Dosage: {dosage}")
                    print(f"Frequency: {frequency}")
                    print(f"Duration: {duration}")
                    print(f"Instructions: {instructions}")
                    print(f"{'='*60}\n")
                    
                    medicines.append({
                        'name': med_name,
                        'dosage': dosage,
                        'frequency': frequency,
                        'timing': parse_timing_from_frequency(frequency),
                        'duration': duration,
                        'instructions': instructions
                    })
        
        i += 1
    
    # Remove duplicates
    seen = set()
    unique_medicines = []
    for med in medicines:
        key = med['name'].lower()
        if key not in seen:
            seen.add(key)
            unique_medicines.append(med)
    
    print(f"\nTotal medicines extracted: {len(unique_medicines)}\n")
    return unique_medicines

def extract_duration_from_block(text):
    """
    Extract duration from a complete medicine block.
    Tries multiple patterns to catch all variations.
    """
    print(f"  [Duration Search] Searching in: {text}")
    
    # Remove extra spaces and normalize
    text = ' '.join(text.split())
    
    # Pattern 1: "For 3 days" or "For 3 Oays" (OCR sometimes misreads)
    patterns = [
        r'[Ff]or\s+(\d+)\s*([Dd]ays?|[Oo]ays?|weeks?|months?)',  # For 3 days, For 3 Oays
        r'(\d+)\s*([Dd]ays?|[Oo]ays?|weeks?|months?)',  # 3 days, 3 Oays
        r'[Dd]uration\s*:?\s*(\d+)\s*([Dd]ays?|weeks?|months?)',  # Duration: 3 days
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            number = match.group(1)
            unit = match.group(2).lower()
            
            # Normalize "oays" to "days" (common OCR mistake)
            if 'oay' in unit:
                unit = 'days'
            
            result = f"{number} {unit}"
            print(f"  [Duration Found] {result}")
            return result
    
    print(f"  [Duration Not Found] Using default: 7 days")
    return '7 days'

def extract_dosage_from_text(text):
    """Extract dosage information"""
    dosage_match = re.search(r'(\d+\.?\d*\s*(?:mg|ml|g|mcg|tab|tablets?|cap|capsules?))', text, re.IGNORECASE)
    return dosage_match.group(1) if dosage_match else 'As directed'

def extract_frequency_from_text(text):
    """Extract frequency (e.g., 1-0-1, twice daily, once at night)"""
    # Pattern for "once at night", "once in morning", etc
    specific_time_pattern = re.search(r'([Oo]nce)\s+(?:at|in|before|after)?\s*(night|evening|morning|afternoon|bedtime)', text, re.IGNORECASE)
    if specific_time_pattern:
        time_of_day = specific_time_pattern.group(2).lower()
        return f"Once at {time_of_day}"
    
    # Pattern for "twice daily", "once daily", etc
    freq_pattern = re.search(r'((?:[Oo]nce|[Tt]wice|[Tt]hrice|\d+\s*times?)\s*(?:daily|per day|a day))', text, re.IGNORECASE)
    if freq_pattern:
        return freq_pattern.group(1)
    
    # Pattern for 1-0-1 style
    freq_pattern1 = re.search(r'(\d+\s*-\s*\d+\s*-\s*\d+)', text)
    if freq_pattern1:
        return freq_pattern1.group(1)
    
    # Pattern for "BD", "TDS", "QID", "OD"
    freq_pattern3 = re.search(r'\b(OD|BD|TDS|QID)\b', text, re.IGNORECASE)
    if freq_pattern3:
        return freq_pattern3.group(1).upper()
    
    return 'As directed'

def extract_instructions_from_text(text):
    """Extract special instructions"""
    instructions = []
    
    if re.search(r'[Bb]efore\s+(?:food|meal|breakfast|lunch|dinner)', text, re.IGNORECASE):
        instructions.append('Before food')
    elif re.search(r'[Aa]fter\s+(?:food|meal|breakfast|lunch|dinner)', text, re.IGNORECASE):
        instructions.append('After food')
    
    if re.search(r'empty\s+stomach', text, re.IGNORECASE):
        instructions.append('On empty stomach')
    
    return ', '.join(instructions) if instructions else 'Take as prescribed'

def parse_timing_from_frequency(frequency):
    """Convert frequency to timing array"""
    freq_lower = frequency.lower()
    
    # Check for specific time mentions
    if 'night' in freq_lower or 'evening' in freq_lower or 'bedtime' in freq_lower:
        return ['21:00']
    elif 'morning' in freq_lower:
        return ['09:00']
    elif 'afternoon' in freq_lower:
        return ['14:00']
    # Once daily or OD (default to morning)
    elif 'once' in freq_lower or freq_lower == 'od':
        return ['09:00']
    # Twice daily or BD
    elif 'twice' in freq_lower or freq_lower == 'bd' or '1-0-1' in freq_lower:
        return ['09:00', '21:00']
    # Three times or TDS
    elif 'thrice' in freq_lower or 'three times' in freq_lower or freq_lower == 'tds' or '1-1-1' in freq_lower:
        return ['09:00', '14:00', '21:00']
    # Four times or QID
    elif 'four times' in freq_lower or freq_lower == 'qid' or '1-1-1-1' in freq_lower:
        return ['08:00', '13:00', '18:00', '22:00']
    else:
        return ['09:00']  # Default to once daily in morning

def extract_doctor_name(text):
    """Extract doctor's name"""
    match = re.search(r'(?i)dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})', text)
    return match.group(0) if match else ''

def extract_hospital_name(text):
    """Extract hospital/clinic name"""
    patterns = [
        r'(?i)([A-Z][a-zA-Z\s]+(?:hospital|clinic|medical center|healthcare))',
        r'(?i)(?:hospital|clinic|medical center|healthcare)[:\s]+([A-Z][a-zA-Z\s]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ''

def extract_date(text):
    """Extract date from text"""
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def extract_diagnosis(text):
    """Extract diagnosis information"""
    diagnosis_match = re.search(r'(?i)(?:diagnosis|dx|diagnosed with)[:\s]+([^\n.]+)', text)
    return diagnosis_match.group(1).strip() if diagnosis_match else ''

def extract_advice(text):
    """Extract doctor's advice"""
    advice_matches = re.findall(r'(?i)(?:advice|recommendation)[:\s]+([^\n.]+)', text)
    return [advice.strip() for advice in advice_matches if advice.strip()]

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

# In-memory database
prescriptions_db = {}
medication_logs = {}
patients_db = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_duration_to_days(duration_str):
    """Convert duration string to days"""
    duration_str = duration_str.lower().strip()
    match = re.search(r'(\d+)\s*(day|week|month|year)', duration_str)
    if not match:
        return 7  # Default to 7 days
    number = int(match.group(1))
    unit = match.group(2)
    if 'day' in unit:
        return number
    elif 'week' in unit:
        return number * 7
    elif 'month' in unit:
        return number * 30
    elif 'year' in unit:
        return number * 365
    return 7

def generate_schedule(medicines, start_date=None):
    """Generate medication schedule based on prescription duration"""
    if start_date is None:
        start_date = datetime.now()
    
    if not medicines:
        return [], 0
    
    max_duration_days = max((parse_duration_to_days(m['duration']) for m in medicines), default=7)
    schedule = []
    
    print(f"\n[SCHEDULE GENERATION]")
    print(f"Medicines to schedule: {len(medicines)}")
    for med in medicines:
        days = parse_duration_to_days(med['duration'])
        print(f"  - {med['name']}: {med['duration']} = {days} days")
    print(f"Max duration: {max_duration_days} days\n")
    
    for day in range(max_duration_days):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime('%Y-%m-%d')
        day_schedule = {
            'date': date_str,
            'day': current_date.strftime('%A'),
            'medications': []
        }
        
        for med in medicines:
            med_duration_days = parse_duration_to_days(med['duration'])
            if day < med_duration_days:
                for time in med.get('timing', ['09:00']):
                    day_schedule['medications'].append({
                        'medicine': med['name'],
                        'dosage': med['dosage'],
                        'time': time,
                        'frequency': med.get('frequency', 'As directed'),
                        'instructions': med.get('instructions', 'As prescribed'),
                        'duration': med['duration'],
                        'remaining_days': med_duration_days - day,
                        'taken': False,
                        'id': f"{date_str}_{time}_{med['name']}"
                    })
        
        if day_schedule['medications']:
            day_schedule['medications'].sort(key=lambda x: x['time'])
            schedule.append(day_schedule)
    
    return schedule, max_duration_days

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_prescription():
    if 'prescription' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['prescription']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the prescription
        prescription_data = extract_prescription_data(filepath)
        
        if 'error' in prescription_data:
            return jsonify(prescription_data), 400
        
        # Add form data (override OCR extracted patient name with user input)
        prescription_data['patient_name'] = request.form.get('patient_name', prescription_data.get('patient_name', 'Unknown Patient'))
        prescription_data['phone_number'] = request.form.get('phone_number', '')
        prescription_data['family_contact'] = request.form.get('family_contact', '')
        prescription_data['image_path'] = filepath
        
        # Generate schedule
        schedule, duration_days = generate_schedule(prescription_data['medicines'])
        
        # Store in database
        prescription_id = f"RX{timestamp}"
        prescriptions_db[prescription_id] = {
            'data': prescription_data,
            'schedule': schedule,
            'duration_days': duration_days,
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'prescription_id': prescription_id,
            'data': prescription_data,
            'schedule': schedule,
            'duration_days': duration_days
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/prescriptions')
def list_prescriptions():
    """List all prescriptions"""
    try:
        prescriptions_list = []
        for rx_id, rx_data in prescriptions_db.items():
            prescriptions_list.append({
                'id': rx_id,
                'patient_name': rx_data['data'].get('patient_name', 'Unknown'),
                'date': rx_data['data'].get('date', ''),
                'medicines_count': len(rx_data['data'].get('medicines', [])),
                'duration_days': rx_data.get('duration_days', 7),
                'created_at': rx_data.get('created_at', '')
            })
        
        prescriptions_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return jsonify(prescriptions_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/schedule/<prescription_id>')
def get_schedule(prescription_id):
    """Get schedule for a specific prescription"""
    if prescription_id not in prescriptions_db:
        return jsonify({'error': 'Prescription not found'}), 404
    
    prescription = prescriptions_db[prescription_id]
    return jsonify({
        'prescription_id': prescription_id,
        'schedule': prescription.get('schedule', []),
        'medicines': prescription.get('data', {}).get('medicines', []),
        'patient_name': prescription.get('data', {}).get('patient_name', ''),
        'duration_days': prescription.get('duration_days', 7)
    })

@app.route('/family_dashboard/<prescription_id>')
def family_dashboard(prescription_id):
    """Get family dashboard data for a prescription"""
    if prescription_id not in prescriptions_db:
        return jsonify({'error': 'Prescription not found'}), 404
    
    prescription = prescriptions_db[prescription_id]
    schedule = prescription.get('schedule', [])
    
    # Calculate statistics
    total_meds = 0
    taken_meds = 0
    
    for day in schedule:
        for med in day.get('medications', []):
            total_meds += 1
            if med.get('taken'):
                taken_meds += 1
    
    adherence_rate = (taken_meds / total_meds * 100) if total_meds > 0 else 0
    
    return jsonify({
        'prescription_id': prescription_id,
        'patient_name': prescription.get('data', {}).get('patient_name', ''),
        'schedule': schedule,
        'statistics': {
            'total_medications': total_meds,
            'taken': taken_meds,
            'missed': total_meds - taken_meds,
            'adherence_rate': round(adherence_rate, 1)
        },
        'logs': medication_logs.get(prescription_id, {})
    })

@app.route('/mark_taken', methods=['POST'])
def mark_medication_taken():
    """Mark a medication as taken/not taken"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        prescription_id = data.get('prescription_id')
        medication_id = data.get('medication_id')
        taken = data.get('taken', True)
        
        if not prescription_id or not medication_id:
            return jsonify({'error': 'Missing required fields'}), 400
            
        if prescription_id not in prescriptions_db:
            return jsonify({'error': 'Prescription not found'}), 404
            
        # Update the schedule
        for day in prescriptions_db[prescription_id]['schedule']:
            for med in day.get('medications', []):
                if med.get('id') == medication_id:
                    med['taken'] = taken
                    med['taken_at'] = datetime.now().isoformat() if taken else None
                    
                    # Log the action
                    if prescription_id not in medication_logs:
                        medication_logs[prescription_id] = {}
                    medication_logs[prescription_id][medication_id] = {
                        'taken': taken,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return jsonify({'success': True, 'medication': med})
        
        return jsonify({'error': 'Medication not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)