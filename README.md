# Sterling Services: S.O.W. Generator

## Overview

The Sterling Services: S.O.W. Generator is a Streamlit-based application designed to automate and streamline the process of creating Statements of Work (SOW). It uses AWS Bedrock with Claude 3.5 Haiku to analyze project information from audio or text inputs and generate comprehensive answers to predefined project questions.

## Features

- Optimized audio transcription using AWS Transcribe with parallel processing
- Automatic analysis after transcription
- Support for multiple file formats (audio and text)
- AI-powered analysis using AWS Bedrock and Claude 3.5 Haiku
- Customizable project questions with instructions
- Real-time progress tracking
- Multiple output formats (TXT, PDF, DOCX)

## Requirements

- Python 3.10+
- AWS Account with access to:
  - AWS Bedrock (Claude 3.5 Haiku model)
  - AWS Transcribe
  - S3 Bucket
- FFmpeg (for audio processing)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sterling-services-sow-generator.git
   cd sterling-services-sow-generator
   ```

2. Install FFmpeg:
   ```
   # On Mac
   brew install ffmpeg
   
   # On Ubuntu
   sudo apt-get install ffmpeg
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure AWS credentials in `.env` file or environment variables:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   AWS_BUCKET_NAME=your_bucket
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload an audio or text file containing project information.

3. The app will automatically:
   - Transcribe audio files (if applicable)
   - Process the content using AWS Bedrock
   - Generate comprehensive answers to project questions
   - Display results with download options

## Customizing Questions

Edit questions through the UI or directly in `Questions.json`. Each question can include:
- Category
- Question text
- Optional instructions for AI processing

## Performance Optimization

The app uses parallel processing for:
- Audio transcription (chunked processing)
- Question analysis (batch processing)
- Result generation

## Contributing

Contributions to improve the Sterling Services: S.O.W. Generator are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

MIT License

Copyright (c) 2024 GenAI Jake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

For any queries or support, please contact @genaijake on X (formerly Twitter).

## Changelog

See the [CHANGELOG.md](CHANGELOG.md) file for details on recent changes and version history.