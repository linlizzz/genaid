#import ollama

#client = ollama.Client()

#model = "openintegrator/poro-34b-chat"

# response = client.chat(model=model, prompt=prompt)


import torch
from transformers import pipeline


# ask the model to evaluate the diagnosis
prompt_summarization = "You are an expert in healthcare and clinical research, responsible for extracting and summarizing the following five aspects from the given patient clinical notes. \
The input is the given patient clinical notes. \
The output includes five parts: \
1. Summarise the patient history and general condition. \
2. Summarise what the doctor examined. \
3. Summarise the doctor's findings from the examination. \
4. Summarise the doctor's diagnosis. \
5. Summarise the treatment plan. \
Please ensure that the output is based on the patient clinical note and do not include the content outside of the input. Just say the question cannot be answered if you don’t find relevant given information or you don’t know. Make the output concise but complete."

example_summarization = "Input: \
Background: 20-year-old woman with a sudden fever 1 day ago, sore throat. Swallowing slightly painful. \
Current status: Yt good. Cor et pulm 0. Tonsils fire engine red, swollen, white plug. Lymph nodes palpated on neck. Ears healthy. CRP 120. \
Plan: Due to high CPR, referral to KNK emergency room, pharyngeal abscess? Invasive disease? \
Diagnosis: J02.9 - Unspecified acute pharyngitis. \
Output: \
1. The patient has had a fever and sore throat for 24 hours, accompanied by pain when swallowing. \
2. The doctor assessed the patient's general condition, listened to the heart and lungs, examined the throat, ears and neck, and measured C-reactive protein. \
3. The tonsils were clearly red and swollen. Enlarged lymph nodes were observed in the neck. The CRP was 120, which is clearly elevated. \
4. The doctor diagnosed the patient with pharyngitis. \
5. The doctor referred the patient to the ENT emergency department."

prompt_evaluation = "You are an expert in healthcare and clinical research, responsible for evaluating whether the doctor's diagnosis and plans are adherence to the given guidelines. You will receive a summary of patient's clinical notes, and guidelines.\
The input includes two parts: \
1. The summary of the patient history and general condition, what the doctor examined, the doctor's findings from the examination, the doctor’s diagnosis, and the treatment plan. \
2. The guideline you should take as a reference. \
The output includes three parts: \
1. Evaluate if the examination strategy follows the given guidelines. Why or why not? \
2. Evaluate if the diagnosis is compatible with the patient history and doctor's findings. \
3. Evaluate if the treatment plan follows the given guidelines. Why or why not? \
Please ensure that the output is based on the guideline and do not include the content outside of the input. Just say the question cannot be answered if you don’t find relevant given information or you don’t know. Make the output concise but complete."

example_evaluation = "Input: \
Summary: 1. The patient has had a fever and sore throat for 24 hours, accompanied by pain when swallowing. \
2. The doctor assessed the patient's general condition, listened to the heart and lungs, examined the throat, ears and neck, and measured C-reactive protein. \
3. The tonsils were clearly red and swollen. Enlarged lymph nodes were observed in the neck. The CRP was 120, which is clearly elevated. \
4. The doctor diagnosed the patient with pharyngitis. \
5. The doctor referred the patient to the ENT emergency department. \
Guideline: Do not measure CRP when viral or streptococcal A (StrA) pharyngitis is suspected. \
Do not collect a throat swab from a patient with mild symptoms (Centor score 0-2) with sore throat. Centor score for pharyngeal calculi: Absence of cough, 1 point; Tenderness and swelling of submandibular lymph nodes, 1 point; Swollen or covered tonsils, 1 point; Fever above 38°C based on history or measured, 1 point. \
Do not use antimicrobials other than penicillin V for acute pharyngitis unless the patient has a contraindication to penicillin. The recommended treatment for recurrent infection is a first-generation cephalosporin. \
Output: \
1. The activity does not follow the recommendation - CRP is unnecessary in the diagnosis of a sore throat patient, as CRP should generally not be tested when pharyngitis is suspected. \
2. The diagnosis of pharyngitis fits the preliminary information and research findings. \
3. The question cannot be answered based on the data provided."

user_input = "Call: See above. 40-year-old man with fever of 39C and sore throat that started 2 days ago. Throat culture now shows S. pyogenes +++. Called, good job, needed painkillers. V-pen mega 1x3 for 10 days is started. Need new contact. \
Diagnosis: J02.0 Streptococcal pharyngitis."

guideline = "Do not measure CRP when viral or streptococcal A (StrA) pharyngitis is suspected. \
Do not collect a throat swab from a patient with mild symptoms (Centor score 0-2) with sore throat. Centor score for pharyngeal calculi: Absence of cough, 1 point; Tenderness and swelling of submandibular lymph nodes, 1 point; Swollen or covered tonsils, 1 point; Fever above 38°C based on history or measured, 1 point. \
Do not use antimicrobials other than penicillin V for acute pharyngitis unless the patient has a contraindication to penicillin. The recommended treatment for recurrent infection is a first-generation cephalosporin."

if __name__ == '__main__':

    query_summarization = user_input
    messages_summarization = [
        {"role": "user", "content": prompt_summarization},
        {"role": "assistant", "content": "Could you give me an example?"},
        {"role": "user", "content": example_summarization},
        {"role": "assistant", "content": "Please tell me the patient's clinical note and the coressponding guideline. I will give the response of my task based on it."},
        {"role": "user", "content": query_summarization}
    ]

    
    pipe = pipeline("text-generation", model="BioMistral/BioMistral-7B", torch_dtype=torch.bfloat16, device="cuda", max_new_tokens=1000)
    
    response_summary_all = pipe(messages_summarization)
    response_summary = response_summary_all[0]['generated_text'][-1]['content']

    print("----BioMistral-7B----\n\n")
    print("<<< Response:", response_summary)
    print("\n\n")

    query_evaluation = response_summary + "\n" + guideline
    messages_evaluation = [
        {"role": "user", "content": prompt_evaluation},
        {"role": "assistant", "content": "Could you give me an example?"},
        {"role": "user", "content": example_evaluation},
        {"role": "assistant", "content": "Please tell me the patient's clinical note and the coressponding guideline. I will give the response of my task based on it."},
        {"role": "user", "content": query_evaluation}
    ]
    
    response_evaluation_all = pipe(messages_evaluation)
    response_evaluation = response_evaluation_all[0]['generated_text'][-1]['content']

    print("<<< Response:", response_evaluation)
    print("\n\n")
