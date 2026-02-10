import os
import textwrap

#Disable HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline


#LOAD MODEL
def load_model():
    print("Loading Question Answering Model...")
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )
    print("Model loaded successfully!\n")
    return qa_pipeline


#FORMAT OUTPUT
def format_output(result):
    answer = result.get("answer", "").strip()
    score = round(result.get("score", 0) * 100, 2)

    print("\n" + "=" * 50)
    if answer:
        print("ANSWER:")
        print(textwrap.fill(answer, width=70))
        print(f"\nCONFIDENCE SCORE: {score}%")
    else:
        print("ANSWER: No suitable answer found.")
        print("CONFIDENCE SCORE: 0%")
    print("=" * 50 + "\n")


#MAIN APP
def run_qna_app():
    qa_model = load_model()

    print("Question Answering System (Transformer-based)")
    print("Type 'exit' anytime to quit.\n")

    while True:
        # Take context
        context = input("Enter Context:\n").strip()
        if context.lower() == "exit":
            print("Exiting application. Bye")
            break
        if not context:
            print("Context cannot be empty. Please try again.\n")
            continue

        # Take question
        question = input("\nEnter Question:\n").strip()
        if question.lower() == "exit":
            print("Exiting application. Bye")
            break
        if not question:
            print("Question cannot be empty. Please try again.\n")
            continue

        try:
            result = qa_model(question=question, context=context)
            format_output(result)

        except Exception as e:
            print("Error occurred:", str(e))


#RUN
if __name__ == "__main__":
    run_qna_app()