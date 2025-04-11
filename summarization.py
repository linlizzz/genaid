from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    return summarizer(text, max_length=130, min_length=40, do_sample=False)


if __name__ == "__main__":
    text = "Cause of result: Cough\
Preliminary information: 32-year-old generally healthy woman. General respiratory symptoms for 5-6 days, \
    now cough is emphasized, mild flank pain with coughing fits. No fever, ear or sinus symptoms. \
Current status:  Health good. Breathing free, calm. Heart rate steady, no rhythm. RR 127/80 p 70. \
    Mild mucus rales in the lungs. Ttymp 36.5. Spo2 98%. Rapid CRP 16. \
Plan: Impression of viral bronchitis, symptomatic treatment. Speech work: sva 3 days, new appointment required. \
Diagnosis: J20.9 Unspecified acute bronchitis."
    print("<<< Summary text:", summarize(text))
