# 📌 RL-diabetes
*A Reinforcement Learning (RL) agent for personalized daily exercise & therapy planning in diabetes management.*  

---

## 📖 Overview
Diabetes management often requires careful planning of **daily physical activity** and **therapy adherence**.  
This project explores **Reinforcement Learning (RL)** to generate **personalized recommendations** that optimize both **adherence** and **health outcomes**.  

- ⚙️ **Language**: Python 3.10  
- 📦 **Environment**: Dockerized for reproducibility  
- 🖥️ **Editor**: VS Code  

---

## 📂 Project Structure
```
RL-diabetes/
│── notebooks/        # Jupyter notebooks for experiments
│── src/              # Core RL source code
│── tests/            # Unit tests
│── main.py           # Entry point for running the project
│── requirements.txt  # Python dependencies
│── Dockerfile        # Docker environment setup
│── .dockerignore     # Ignore unnecessary files in Docker
│── .gitignore        # Ignore unnecessary files in Git
│── README.md         # Documentation
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/chen33001/RL-diabetes.git
cd RL-diabetes
```

---

### 2. Local Setup (Python 3.10)

Create virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the project:
```bash
python main.py
```

---

### 3. Docker Setup

Build the Docker image:
```bash
docker build -t rl-diabetes .
```

Run the container:
```bash
docker run -it --rm rl-diabetes
```

Run Jupyter Notebook inside Docker:
```bash
docker run -it -p 8888:8888 rl-diabetes \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Then open [http://localhost:8888](http://localhost:8888) in your browser.

---

## 🧪 Testing
```bash
pytest tests/
```

---

## 📊 Roadmap
- [ ] Define environment for diabetes daily planning  
- [ ] Implement RL agent (baseline Q-learning)  
- [ ] Extend with Deep RL (DQN, PPO)  
- [ ] Integrate smart device data (steps, heart rate, glucose)  
- [ ] Deploy with API (FastAPI/Flask)  

---

## 🤝 Contributing
1. Fork the project  
2. Create a feature branch (`git checkout -b feature/new-feature`)  
3. Commit your changes (`git commit -m "feat: add new feature"`)  
4. Push to the branch (`git push origin feature/new-feature`)  
5. Open a Pull Request  

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
