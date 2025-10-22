# RL-diabetes

A reinforcement learning agent that proposes daily exercise plans for people living with diabetes. It ships with a custom Gymnasium environment, PPO training script, notebooks for experiments, and CI workflows.

---

## Overview
- Language: Python 3.12
- RL stack: Stable Baselines3 (PPO) + Gymnasium
- Tooling: Docker image, pytest suite, GitHub Actions

---

## Project Structure
```
RL-diabetes/
|- notebooks/         # Exploratory notebooks and experiments
|- src/               # Custom Gymnasium environments and utilities
|- tests/             # Unit tests (pytest)
|- main.py            # Entry point for running the PPO agent
|- requirements.txt   # Python dependencies
|- Dockerfile         # Containerised runtime environment
|- .github/workflows/ # CI pipeline definition
|- README.md          # Project documentation
```

---

## Getting Started (Python 3.12)
```bash
git clone https://github.com/chen33001/RL-diabetes.git
cd RL-diabetes

python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt

python main.py                   # trains PPO on the diabetes environment
```

> **PyTorch wheels for Python 3.12**  
> If PyTorch is unavailable for your platform by default, install it first by following the matrix at https://pytorch.org/get-started/locally/ (select Python 3.12), then rerun `pip install -r requirements.txt`.

---

## Docker Workflow
```bash
docker build -t rl-diabetes .
docker run -it --rm rl-diabetes
```

To expose Jupyter Notebook from the container:
```bash
docker run -it -p 8888:8888 rl-diabetes \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```
Open http://localhost:8888 and use the token printed in the container logs.

---

## Testing
```bash
pytest
```

---

## Roadmap
- [ ] Broaden agent portfolio (DQN, SAC)
- [ ] Enrich environment with nutrition and insulin dynamics
- [ ] Integrate wearable-device data loaders
- [ ] Provide FastAPI service for inference

---

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your work (`git commit -m "feat: add my feature"`).
4. Push the branch (`git push origin feature/my-feature`).
5. Open a pull request.

---

## License
This project is released under the [MIT License](LICENSE).
