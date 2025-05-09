#import ollama

#client = ollama.Client()

#model = "openintegrator/poro-34b-chat"

# response = client.chat(model=model, prompt=prompt)

import numpy as np
import torch
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Test if the model can understand the domain knowledge

prompt = "Olet kokenut lääkäri ja sinun on annettava diagnoosi ja hoitosuunnitelma kyselystä annettujen kliinisten huomautusten mukaisesti."

user_input = "Esitiedot: 62-v nainen, taustalla RR-tauti, hyperkolesterolemia ja DM2. Obesiteettia. Lääkityksenä: Metformiini 500mg 2+2, kandesartaani 8mg x1 ja atorvastatiini 40mg, estrogeenikorvaushoito. Tupakoi. Suvussa sydän- ja verisuonitauteja: äidille tehty pallonlaajennus 62- vuotiaana. Asuu puolison kanssa, työelämässä.\
Seurattu diabeteshoitajalla kerran vuodessa, ed lääkärin kontrolli 3v sitten. Tällöin tarkastettu silmänpohjakuvat jotka kunnossa, ei retinopatiaan sopivia muutoksia. Samaten krea normaali ja jalkojen tunnot normaalit. EI kroonisia haavoja anamneesissa. \
Nykytila: Yt hyvä. Sydämestä ja keuhkoista ei ausk. poikkeavaa. Iho siisti, ei haavoja. Monofilamentilla jalkapohjien terävätunto hieman heikentynyt, jotain tuntee. ADP +, ATP+, jalat hieman viileät. Vatsanpeitteet runsaat. 90kg, 160cm. Verenpaineet kotimittauksin 140-155/87-90 viikon seurannassa. \
Lab: PVK viitteissä, Krea 103, josta GFR 62, PLV siisti, ei proteiinia. LDL 4, HDL 1.1, trigly 2,8, HbAIc 52, fP-gluk 7.0. Ei kotimittauksia.\
Suunnitelma: \
BM 35, toivoo laihtuvansa. Aloitetaan ozempic: Annosnostot kuukausittain.\
Diabeteksen hoitotasapaino melko hyvä, mutta silti viitteitä neuropatiasta -> tehostetaan lääkitys ( ozempic) ja kontrolli lääkärille 1 v päähän. Silmänpohjakuviin ohjattu, lisäksi U-alb ja U-alb/krea jo nyt. \
Verenpaineet melko hyvällä tasolla. Seuraa kuitenkin ettei lähde tuosta nousemaan. Jatkosssa kandesartaani tehostus.\
Potilaalla ei sydän ja verisuonitauteja mutta merkittävä sukuriski + tupakoi. LDL koholla, nostetaan atorvastatiiniannosta ad 80mg. Soitto voinnista 2kk päähän. Jos voimakkaita lihashaittoja niin yhteydessä ja tällöin rabdomyolyysin poissulku.\
Kannustettu tupakoinnin lopettamiseen. Champix erec.\
Metformiiniannosta ei voida nostaa, ei siedä suurempaa annosta."





# ask the model to evaluate the diagnosis
prompt_summary = "Olet terveydenhuollon ja kliinisen tutkimuksen asiantuntija, jonka vastuulla on poimia ja tiivistää seuraavat viisi näkökohtaa annetuista potilaan kliinisistä tiedoista. \
Syöte on annettu potilaan kliininen muistiinpano. \
Tuloste sisältää viisi osaa: \
1. Yhteenveto potilaan historiasta ja yleiskunnosta. \
2. Yhteenveto lääkärin tutkimista asioista. \
3. Yhteenveto lääkärin löydöksistä tutkimuksessa. \
4. Yhteenveto lääkärin diagnoosista. \
5. Yhteenveto hoitosuunnitelmasta. \
Varmista, että tuloste perustuu potilaan kliiniseen muistiinpanoon, äläkä sisällytä syötteen ulkopuolista sisältöä. Sano vain, että kysymykseen ei voida vastata, jos et löydä asiaankuuluvaa tietoa annetuista tiedoista tai et tiedä. Tee tuotoksesta ytimekäs mutta kattava.\
Esimerkki: \
Syöte: \
Esitiedot: 20-v nainen, jolla 1 vrk sitten äkillisesti noussut kuume, kurkku karhea. Nieleminen hiukan kivuliasta. \
Nykytila: Yt hyvä. Cor et pulm 0. Nielurisat paloautonpunaiset, turvonneet, valkeaa proppua. Kaulalta palp imusolmukkeita. Korvat terveet. CRP 120. \
Suunnitelma: Korkean CPR:n vuoksi lähete KNK-päivystykseen, nielupaise? Invasiivinen tauti? \
Diagnoosi: J02.9 - Määrittämätön akuutti nielutulehdus. \
Tulos: \
1. Potilaalla on ollut kuumetta ja kurkkukipua 24 tuntia, johon liittyy kipua niellessä. \
2. Lääkäri arvioi potilaan yleistilan, kuunteli sydäntä ja keuhkoja, tutki kurkun, korvat ja kaulan sekä mittasi C-reaktiivisen proteiinin. \
3. Nielurisat olivat selvästi punaiset ja turvonneet. Kaulassa havaittiin suurentuneita imusolmukkeita. CRP oli 120, mikä on selvästi koholla. \
4. Lääkäri diagnosoi potilaalla nielutulehduksen. \
5. Lääkäri lähetti potilaan korva-, nenä- ja kurkkutautien ensiapuun."


prompt_evaluation = "Olet terveydenhuollon ja kliinisen tutkimuksen asiantuntija, jonka vastuulla on arvioida, ovatko lääkärin diagnoosi ja suunnitelmat annettujen ohjeiden mukaisia. Saat yhteenvedon potilaan kliinisistä tiedoista. \
Syöte sisältää viisi osaa: \
1. Potilaan historia ja yleistila. \
2. Mitä lääkäri tutki. \
3. Lääkärin löydökset tutkimuksesta. \
4. Lääkärin diagnoosi. \
5. Hoitosuunnitelma. \
Tuloste sisältää kolme osaa: \
1. Arvioi, noudattaako tutkimusstrategia annettuja ohjeita. Miksi tai miksi ei? \
2. Arvioi, onko diagnoosi yhteensopiva potilaan historian ja lääkärin löydösten kanssa. \
3. Arvioi, noudattaako hoitosuunnitelma annettuja ohjeita. Miksi tai miksi ei? \
Varmista, että tuloste perustuu ohjeisiin äläkä sisällytä syötteen ulkopuolista sisältöä. Sano vain, että kysymykseen ei voida vastata, jos et löydä olennaisia ​​tietoja tai et tiedä. Tee tulosteesta ytimekäs mutta kattava."


user_input = "Esitiedot: 16-v poika, jolla 2-3 vrk ajan nuhaa, lievää yskää, kurkkukipua. Äiti epäilee angiinaa. \
Nykytila: Äidin kanssa vo:lla. Yt hyvä, hengitys vapaata rauhallista. Sydämestä ja keuhkoista ei kuulu poikkeavaa. Kaulalta ei palpoidu poikkeavaa. Nielu diffuusisti punoittaa, risat kenties hieman turvonneet. \
Suunnitelma: Streptokokki-infektion poissulkuun nieluviljely, tarkistavat vastauksen itse OmaKannasta. Muuten oireenmukainen hoito. \
Diagnoosi: J06.9 Määrittämätön akuutti ylähengitystieinfektio."


def create_guideline_embeddings(guidelines):
    # Split guidelines into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_text(guidelines)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    # Create vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_relevant_guidelines(vectorstore, query, k=5):
    # how to get the number of k accordingly?

    # Search for relevant guidelines
    relevant_docs = vectorstore.similarity_search(query, k=k)

    guidelines = "\n".join([doc.page_content for doc in relevant_docs])
    
    # what's being retrieved
    print("\n=== Retrieved Guidelines ===")
    print(f"Number of chunks retrieved: {len(relevant_docs)}")
    print("\nGuidelines content:")
    print(guidelines)
    print("========================\n")

    return "\n".join([doc.page_content for doc in relevant_docs])


def finnish_llm(model, user_input, prompt, guideline_store, setting="evaluation"):
    # Get relevant guidelines for the query
    relevant_guidelines = get_relevant_guidelines(guideline_store, user_input)
    
    # Combine prompt with relevant guidelines
    augmented_prompt = f"{prompt_evaluation}\n\nRelevant guidelines:\n{relevant_guidelines}"
    
    query = user_input
    pipe = pipeline("text-generation", 
                    model=model, 
                    torch_dtype=torch.bfloat16, 
                    device="cuda", 
                    max_new_tokens=1000)
    
    if setting == "evaluation":
        messages = [
            {"role": "system", "content": augmented_prompt},
            {"role": "user", "content": query}
        ]   
    elif setting == "summary":
        messages = [
            {"role": "system", "content": prompt_summary},
            {"role": "user", "content": query}
        ]
    response_all = pipe(messages)
    response= response_all[0]['generated_text'][-1]['content']
    print("<<< Response:\n\n", response, "\n\n<<<End of Response\n\n")
    return response

if __name__ == '__main__':
    # models = ["LumiOpen/Poro-34B-chat", "BioMistral/BioMistral-7B", "utter-project/EuroLLM-9B-Instruct"]

    # Create guideline embeddings
    guideline = open("data/guideline.txt", "r").read()  
    guideline_store = create_guideline_embeddings(guideline)
    
    response_summary = finnish_llm(model="LumiOpen/Poro-34B-chat", 
                           user_input=user_input, 
                           prompt=prompt_summary, 
                           guideline_store=guideline_store, 
                           setting="summary")
    
    response_evaluation = finnish_llm(model="LumiOpen/Poro-34B-chat", 
                           user_input=response_summary, 
                           prompt=prompt_evaluation, 
                           guideline_store=guideline_store, 
                           setting="evaluation")
    

