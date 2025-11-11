# 🎬 Movie Title Next-Word Prediction App (LSTM +  Streamlit)

This project generates the **next possible words for movie titles** using a **Recurrent Neural Network (RNN)** built with **Bidirectional LSTM**. It is trained on the **TMDB 5000 Movie Dataset**, and provides a **Streamlit-based UI** to generate creative, movie-like titles in real time.

---

## 🚀 Features

* Learns from **5000+ real movie titles**
* Uses **Tokenizer + Word Embeddings** to understand text
* **Bidirectional LSTM** to learn context from multiple directions
* Easy-to-use **Streamlit interface**
* **On-demand model training** feature
* Saves trained models for reuse

---

## 🧠 Model Architecture

```
Embedding Layer (vocab_size, 32)
Bidirectional LSTM (128 units, return_sequences=True)
Bidirectional LSTM (128 units)
Dense (256 units, ReLU activation)
Dropout (0.3)
Dense (vocab_size, Softmax)
```

This architecture helps the model understand both **forward and backward dependencies** between words, improving prediction quality.

---

## 📂 Dataset

Dataset Used: **TMDB 5000 Movies Dataset**

Download here: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

The model trains only on the `original_title` column.

---

## 🛠️ Installation & Setup


### ** Create Virtual Environment (Required)**

```bash
python -m venv nwp-env
```

### **3️ Activate Environment**

Windows:

```bash
nwp-env\Scripts\activate
```

### **4️ Install Dependencies**

```bash
pip install -r requirements.txt
```

> **Important:** Use **Python 3.10** for TensorFlow compatibility.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. Run the Streamlit app.
2. Type a **starting word** (must exist in dataset).
3. Select how many words to generate.
4. Click **Generate**.

### Example Valid Words

```
avatar
cloudy
mars
harry
mission
star
dark
night
love
prince
```

---

## 🧪 Example Output

| Input    | Generated Title                         |
| -------- | --------------------------------------- |
| `cloudy` | cloudy with a chance of destiny forever |
| `mars`   | mars attacks the final empire           |
| `avatar` | avatar the ancient power awakens        |

---



---

## 🤖 Future Enhancements

* Use **GloVe/Word2Vec embeddings** for deeper semantic understanding
* Add **Top-3 next-word suggestions** instead of one
* Deploy to **Streamlit Cloud / Hugging Face Spaces / Render**
* Add ability to train using **custom movie names**

---



---

## 📝 License

This project is licensed under the **MIT License**.

