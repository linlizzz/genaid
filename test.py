print("work!")

import torch
from transformers import pipeline

prompt = "You are an expert in healthcare and clinical research, responsible for summarising the doctor's diagnosis. \
The input is medical diagnosis record. \
Your output is Summarising the doctor’s diagnosis. \
Example: \
Input: \
Cause of result: Cough. \
Preliminary information: 32-year-old generally healthy woman. General respiratory symptoms for 5-6 days, now cough is emphasized, mild flank pain with coughing fits. No fever, ear or sinus symptoms. \
Current status:  Health good. Breathing free, calm. Heart rate steady, no rhythm. RR 127/80 p 70. Mild mucus rales in the lungs. Ttymp 36.5. Spo2 98%. Rapid CRP 16. \
Plan: Impression of viral bronchitis, symptomatic treatment. Speech work: sva 3 days, new appointment required. \
Diagnosis: J20.9 Unspecified acute bronchitis. \
Output: \
The doctor's diagnosis was viral bronchitis."

user_input = "Interim assessment: 1 week ago flu symptoms, now gone to the lungs. Cor et pulm 0. SpO2 99%. CRP 22. Erec Amorion. \
Diagnosis:J20.9 Unspecified acute bronchitis."



if __name__ == '__main__':


    query = user_input
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    
    pipe = pipeline("text-generation", model="utter-project/EuroLLM-9B-Instruct", torch_dtype=torch.bfloat16, device="cuda", max_new_tokens=1000)
    
    response = pipe(messages)
        #response = client.generate(model=model, prompt=user_input)
    print("----EuroLLM-9B-Instruct----\n\n", "<<< Response: ", response)
    # print("<<< Response: ", response)response)