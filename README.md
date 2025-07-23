# ChatML

ChatML is a compact system that uses on scikit-learn documentation  to answer questions or execute instructions — without agents, LangChain, or orchestration frameworks.

Built with:
- Gemini 2.5 Pro or any language model of your choice 
- scikit-learn docs as context
- Python

Ask a question. If it’s answerable, it answers. If it requires action, it runs it.

---

## Why

To show that LLM workflows don’t need 10 layers of abstraction.  
Direct calls, structured outputs, and real results — that’s it.

---

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## Example prompts
- “Train a decision tree on the data”
- “What hyperparameters can I use for SVM?”
- “Run a logistic regression with regularization”