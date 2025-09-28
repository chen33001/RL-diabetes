📌 RL-diabetes

A Reinforcement Learning (RL) agent for personalized daily exercise & therapy planning in diabetes management.

📖 Overview

Diabetes management often requires careful planning of daily physical activity and therapy adherence.
This project explores Reinforcement Learning (RL) to generate personalized recommendations that optimize both adherence and health outcomes.

⚙️ Language: Python 3.10

📦 Environment: Dockerized for reproducibility

🖥️ Editor: VS Code

📂 Project Structure
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

⚡ Getting Started
1. Clone the repository
git clone https://github.com/chen33001/RL-diabetes.git
cd RL-diabetes

2. Local Setup (Python 3.10)
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt


Run the project:

python main.py

3. Docker Setup

Build the Docker image:

docker build -t rl-diabetes .


Run the container:

docker run -it --rm rl-diabetes


Run Jupyter Notebook inside Docker:

docker run -it -p 8888:8888 rl-diabetes \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root


Then open http://localhost:8888
 in your browser.

🧪 Testing
pytest tests/

📊 Roadmap

 Define environment for diabetes daily planning

 Implement RL agent (baseline Q-learning)

 Extend with Deep RL (DQN, PPO)

 Integrate smart device data (steps, heart rate, glucose)

 Deploy with API (FastAPI/Flask)

🤝 Contributing

Fork the project

Create a feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -m "feat: add new feature")

Push to the branch (git push origin feature/new-feature)

Open a Pull Request

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.
