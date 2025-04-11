#import ollama

#client = ollama.Client()

#model = "openintegrator/poro-34b-chat"

# response = client.chat(model=model, prompt=prompt)


import torch
from transformers import pipeline


# ask the model to evaluate the diagnosis
prompt = "You are an expert in healthcare and clinical research, responsible for evaluating whether the doctor's diagnosis and plans are adherence to the given official guidelines. You will receive a patient's clinical record as the query, and guidelines.\
The input will include two parts:\
1. Query - Medical diagnosis record. Contains the patient's basic information, examination results, doctor's diagnosis and suggestions.\
2. Guidelines. Contains suggestions and contraindications for different disease conditions.\
Your output should include the following parts:\
1. Summarise the patient history. \
2. Summarise what the doctor examined. \
3. Summarise the doctor’s findings. \
4. Summarise the doctor’s diagnosis. \
5. Summarise the treatment plan. \
6. Evaluate if the examination strategy you summarised in 2 follows the given guidelines. Why or why not? \
7. Evaluate if the diagnosis you summarised in 4 is compatible with the patient history and doctor’s findings. \
8. Evaluate if the treatment plan you summarised in 5 follows the given guidelines. Why or why not? \
9. Highlight any necessary guidelines that the doctor has not addressed in the treatment plan. Skip this part if not. \
Please ensure that the output is based on the guidelines and do not introduce suggestions outside of the guidance. Just say the question cannot be answered if you don’t find relevant given information or you don’t know."

example = "Input: \
Query - Medical diagnosis record: \
Cause of result: Cough. \
Preliminary information: 32-year-old generally healthy woman. General respiratory symptoms for 5-6 days, now cough is emphasized, mild flank pain with coughing fits. No fever, ear or sinus symptoms. \
Current status:  Health good. Breathing free, calm. Heart rate steady, no rhythm. RR 127/80 p 70. Mild mucus rales in the lungs. Ttymp 36.5. Spo2 98%. Rapid CRP 16. \
Plan: Impression of viral bronchitis, symptomatic treatment. Speech work: sva 3 days, new appointment required. \
Diagnosis: J20.9 Unspecified acute bronchitis. \
Guidelines:\
Do not use antibiotics to treat bronchitis without specific reasons. \
Acute bronchitis is a disease limited to the mucous membranes of the bronchi, most often caused by a virus. Antibiotic treatment may slightly shorten the duration of symptoms in acute bronchitis. The cough phase of acute bronchitis can last about three weeks. Recovery is individual. The use of antibiotics increases antibiotic resistance. \
The likelihood of pneumonia increases if the patient has the following symptoms: \
fever ≥ 37.8 ºC. \
Increased breathing rate (> 16/min). \
tachycardia (over 95/min). \
oxygen saturation < 96% in room air. \
In addition, the following symptoms indicate pneumonia: \
The illness has affected my general condition. \
The symptoms have developed rapidly. \
The symptoms of the respiratory infection have worsened again. \
The patient has additional risk factors (age, other diseases). \
Sudden bronchitis can turn into pneumonia through a mixed infection, especially in the elderly. \
Output: \
1.The patient has had upper respiratory tract symptoms such as cough and mild flank pain for a few days. He does not have a fever or symptoms of the ears or sinuses. \
2.The doctor performed a clinical examination, which included auscultation of the heart and lungs. In addition, the doctor measured blood pressure, fever, oxygen saturation and inflammation level. \
3.Mild rales were heard in the lungs. The inflammation level was slightly elevated. No other abnormal findings were observed in the examination. \
4.The doctor's diagnosis was viral bronchitis. \
5.The doctor recommended symptomatic treatment and wrote a sick leave certificate. \
6.The question cannot be answered because the treatment recommendation given does not address the patient's examination. \
7.The doctor's diagnosis fits the background information and examination findings. \
8.The doctor's plan is in accordance with the treatment recommendation given, because there were no symptoms or findings suggestive of pneumonia and the patient was therefore prescribed antibiotics."

user_input = "Cause of outcome: flu. \
Preliminary information: This is a 35-year-old woman who is generally healthy, no medication for the weather. She has had a flu for 2 weeks now. It started with a sore throat and fever. Now mostly stuffy, pressure in the sinuses. No earache, no more fever. \
Current status: Yt good, sat 98%, hf calm. No abnormalities in the heart, mucous, rough rales in the lungs l.a. basal. Cleans up after coughing. Ears healthy, slightly negative pressure. Throat clean. Palpation of the neck. Reactive enlarged lymph nodes. RHA clean, sinus scan -/-. Temperature 36.5. \
Plan: Still feeling like a viral disease. No indication of a bacterial infection. Symptomatic treatment with Nasonex, acriseu. Additionally, painkillers needed. Recommended to moisten with sesame oil. New contact if condition worsens or does not improve. SVA 3 days. \
Diagnosis: J06.9 Acute upper respiratory infection, unspecified. \
Guideline: \
Do not use antibiotics to treat bronchitis without specific reasons. \
Acute bronchitis is a disease limited to the mucous membranes of the bronchi, most often caused by a virus. Antibiotic treatment may slightly shorten the duration of symptoms in acute bronchitis. The cough phase of acute bronchitis can last about three weeks. Recovery is individual. The use of antibiotics increases antibiotic resistance. \
The likelihood of pneumonia increases if the patient has the following symptoms: \
fever ≥ 37.8 ºC. \
Increased breathing rate (> 16/min). \
tachycardia (over 95/min). \
oxygen saturation < 96% in room air. \
In addition, the following symptoms indicate pneumonia: \
The illness has affected my general condition. \
The symptoms have developed rapidly. \
The symptoms of the respiratory infection have worsened again. \
The patient has additional risk factors (age, other diseases). \
Sudden bronchitis can turn into pneumonia through a mixed infection, especially in the elderly. "



if __name__ == '__main__':

    query = user_input
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand my role and my task. Could you give me an example?"},
        {"role": "user", "content": example},
        {"role": "assistant", "content": "I know what I should output now. Please tell me the patient's clinical record and the coressponding guidelines. I will give the response of my task based on it."},
        {"role": "user", "content": query}
    ]

    
    pipe = pipeline("text-generation", model="BioMistral/BioMistral-7B", torch_dtype=torch.bfloat16, device="cuda", max_new_tokens=1000)
    
    response = pipe(messages)

    print("----BioMistral-7B----\n\n")
    print("<<< Response:", response)
