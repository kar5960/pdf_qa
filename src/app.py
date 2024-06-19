from flask import Flask, request, render_template
from extract_text import extract_text_from_pdf
from train_model import train_word2vec_model
from answer_question import load_model, get_answer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf = request.files["pdf"]
        question = request.form["question"]
        pdf_path = "data/uploaded.pdf"
        pdf.save(pdf_path)
        
        text = extract_text_from_pdf(pdf_path)
        train_word2vec_model(text)
        
        model = load_model()
        answer = get_answer(model, question, text)
        
        return render_template("index.html", answer=answer)
    return render_template("index.html", answer="")

if __name__ == "__main__":
    app.run(debug=True)