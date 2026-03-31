#!/bin/bash
# ScoreHunt AI Proctoring System - Linux/macOS Launch Script

# 1. Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}[ScoreHunt] Initializing Environment...${NC}"

# 2. Check for Python
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed. Please install Python 3.9+."
    exit 1
fi

# 3. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "${GREEN}[ScoreHunt] Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# 4. Activate Venv
source venv/bin/activate

# 5. Install requirements
echo -e "${GREEN}[ScoreHunt] Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 6. Run Application
echo -e "${GREEN}[ScoreHunt] Launching server...${NC}"
python3 app.py
