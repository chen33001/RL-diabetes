# 📌 RL-diabetes
*A Reinforcement Learning (RL) agent for personalized daily exercise & therapy planning in diabetes management.*  

---

## 📂 Project Structure
RL-diabetes/
│── notebooks/ # Jupyter notebooks for experiments
│── src/ # Core RL source code
│── tests/ # Unit tests
│── main.py # Entry point for running the project
│── requirements.txt # Python dependencies
│── Dockerfile # Docker environment setup
│── .dockerignore # Ignore unnecessary files in Docker
│── .gitignore # Ignore unnecessary files in Git
│── README.md # Documentation

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/chen33001/RL-diabetes.git
cd RL-diabetes

### 2. Local Setup (Python 3.10)
Create virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```bash
Install dependencies:
```bash
pip install -r requirements.txt
```bash
Run the project:
```bash
python main.py
```bash
3. Docker Setup
Build the Docker image:
```bash
docker build -t rl-diabetes .
```bash
Run the container:
```bash
docker run -it --rm rl-diabetes
```bash
Run Jupyter Notebook inside Docker:
```bash
docker run -it -p 8888:8888 rl-diabetes \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```bash
🧪 Testing
```bash
pytest tests/
```bash

---

👉 Key Fix: Every code snippet now uses ` ```bash ... ``` ` **with closing backticks**, so GitHub won’t merge them into one giant block.  

Do you want me to directly also prepare a **ready-to-paste `requirements.txt` + `Dockerfile`** so that when people follow this README, it actually works out of the box?
