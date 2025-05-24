#!/usr/bin/env python3
# scripts/test_transcription.py
"""Test script to transcribe Hindi audio files from the dataset using the ASR API.
"""
import os
import glob
import requests
import json
import csv
from pathlib import Path
import logging
import time
import argparse

# Import the convert_to_wav function from our existing script
from convert_mp3_to_wav import convert_to_wav

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/transcribe"
HINDI_DATASET_PATH = "/home/sarthak-bhardwaj/.cache/kagglehub/datasets/hmsolanki/indian-languages-audio-dataset/versions/1/Indian_Languages_Audio_Dataset/Hindi"
TEMP_DIR = Path("./temp_audio")

def transcribe_audio(wav_file):
    """Send audio file to ASR API for transcription."""
    try:
        with open(wav_file, "rb") as f:
            files = {"file": (os.path.basename(wav_file), f, "audio/wav")}
            response = requests.post(API_URL, files=files, timeout=30)  # 30 second timeout
            
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error transcribing {wav_file}: {e}")
        return None

def decode_unicode_text(text):
    """Convert Unicode Hindi text to readable form."""
    # The \u2581 character is a special token used by NeMo models to represent word boundaries
    return text.replace('\u2581', ' ').strip()

def save_to_csv(results, output_file):
    """Save transcription results to a CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'decoded_text', 'processing_time_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'file_name': result['file_name'],
                'decoded_text': result.get('decoded_text', ''),
                'processing_time_ms': result['transcription'].get('processing_time_ms', 0)
            })

def main():
    """Main function to test transcription on Hindi audio files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test ASR transcription on Hindi audio files')
    parser.add_argument('--num-files', type=int, default=5, help='Number of files to process (default: 5)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json',
                        help='Output format for results (default: json)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    args = parser.parse_args()
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get a list of MP3 files from the Hindi dataset
    mp3_files = glob.glob(os.path.join(HINDI_DATASET_PATH, "*.mp3"))[:args.num_files]
    
    if not mp3_files:
        logger.error(f"No MP3 files found in {HINDI_DATASET_PATH}")
        return
    
    logger.info(f"Found {len(mp3_files)} MP3 files for testing")
    
    results = []
    
    # Process each MP3 file one at a time
    for index, mp3_file in enumerate(mp3_files, 1):
        file_name = os.path.basename(mp3_file)
        wav_file = str(TEMP_DIR / file_name.replace(".mp3", ".wav"))
        
        logger.info(f"\nProcessing file {index}/{len(mp3_files)}: {file_name}")
        logger.info("-" * 50)
        
        # Convert MP3 to WAV using our existing function
        logger.info(f"Converting {file_name} to WAV format...")
        if convert_to_wav(mp3_file, wav_file, target_sr=16000, channels=1):
            # Transcribe WAV file
            logger.info(f"Transcribing {wav_file}...")
            start_time = time.time()
            transcription = transcribe_audio(wav_file)
            end_time = time.time()
            
            if transcription:
                # Add processing time if not provided by API
                if 'processing_time_ms' not in transcription:
                    transcription['processing_time_ms'] = round((end_time - start_time) * 1000, 2)
                
                # Get the transcription text and decode it
                # The API response structure has 'transcription' nested inside the response
                transcription_text = transcription.get('transcription', '')
                if not transcription_text and isinstance(transcription, dict):
                    # Check if it's in a different format
                    transcription_text = transcription.get('text', '')
                decoded_text = decode_unicode_text(transcription_text)
                
                results.append({
                    "file_name": file_name,
                    "transcription": transcription,
                    "decoded_text": decoded_text
                })
                
                logger.info(f"Raw transcription: {transcription_text}")
                logger.info(f"Decoded text: {decoded_text}")
                logger.info(f"Processing time: {transcription.get('processing_time_ms')} ms")
            else:
                logger.error(f"Failed to transcribe {file_name}")
        else:
            logger.error(f"Failed to convert {file_name} to WAV format")
        
        logger.info("-" * 50)
    
    # Print summary of results
    logger.info("\n" + "="*50)
    logger.info(f"Transcription Results Summary ({len(results)}/{len(mp3_files)} files processed)")
    logger.info("="*50)
    
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['file_name']}")
        logger.info(f"   Decoded Text: {result.get('decoded_text', 'N/A')}")
        logger.info(f"   Processing Time: {result['transcription'].get('processing_time_ms', 'N/A')} ms")
        logger.info("-"*50)
    
    # Save results to files
    if results:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save as JSON if requested
        if args.format in ['json', 'both']:
            json_output_file = os.path.join(args.output_dir, f"transcription_results_{timestamp}.json")
            
            # Prepare data for JSON serialization
            output_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files_processed": len(results),
                "total_files_attempted": len(mp3_files),
                "results": []
            }
            
            for result in results:
                output_data["results"].append({
                    "file_name": result["file_name"],
                    "decoded_text": result.get("decoded_text", ""),
                    "processing_time_ms": result["transcription"].get("processing_time_ms", 0)
                })
            
            # Write to JSON file
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to JSON file: {json_output_file}")
        
        # Save as CSV if requested
        if args.format in ['csv', 'both']:
            csv_output_file = os.path.join(args.output_dir, f"transcription_results_{timestamp}.csv")
            save_to_csv(results, csv_output_file)
            logger.info(f"Results saved to CSV file: {csv_output_file}")
    else:
        logger.warning("No results to save to file.")

if __name__ == "__main__":
    main()
