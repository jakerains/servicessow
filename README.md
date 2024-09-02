# Sterling Services: S.O.W. Generator

## Overview

The Sterling Services: S.O.W. Generator is a Streamlit-based application designed to automate and streamline the process of creating Statements of Work (SOW). It uses OpenAI's GPT models to analyze project information from audio or text inputs and generate comprehensive answers to predefined project questions.

## Features

- Audio transcription using OpenAI's Whisper API
- Text file processing
- Customizable project questions
- AI-powered analysis of project information
- Generation of formatted results in TXT, PDF, and DOCX formats
- Downloadable transcripts with timestamps

## Requirements

- Python 3.7+
- Streamlit
- OpenAI Python library
- Other dependencies (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sterling-services-sow-generator.git
   cd sterling-services-sow-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable or be prepared to enter it in the app.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your web browser.

3. Enter your OpenAI API key and select the GPT model you want to use.

4. Upload an audio or text file containing project information.

5. Customize the project questions if needed.

6. Click "Analyze" to process the input and generate the SOW.

7. Download the results in your preferred format (TXT, PDF, or DOCX).

## Customizing Questions

The app allows you to customize the questions used for SOW generation. You can add, edit, or delete questions through the user interface. Changes are automatically saved to `Questions.json`.

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