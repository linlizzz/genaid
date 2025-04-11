#import ollama

#client = ollama.Client()

#model = "openintegrator/poro-34b-chat"

# response = client.chat(model=model, prompt=prompt)


import torch
from transformers import pipeline


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
prompt = "You are an expert in healthcare and clinical research, responsible for evaluating whether the doctor's diagnosis and plans are adherence to the given official guidelines. You will receive a patient's clinical record as the query, and guidelines.\
The input will include two parts:\
1. Query - Medical diagnosis record. Contains the patient's basic information, examination results, doctor's diagnosis and suggestions.\
2. Guidelines. Contains suggestions and contraindications for different disease conditions.\
Your tasks are:\
1. Analyze the doctor's diagnosis and plans.\
2. Compare the guidelines to determine whether the doctor's diagnosis and plans are adherence to the guidelines.\
3. Indicate which parts are adherence to the guidelines and which parts are not.\
4. Indicate the suggestions in the guidelines that the doctor did not mention.\
Your output should include the following parts:\
1. Validation Result containing diagnosis alignment and plan alignment. Indicate whether the doctor’s diagnosis and plans align with the guidelines.\
2. Detailed Report containing compliance, non-compliance, and omissions.\
Compliance: List the parts of the doctor's diagnosis and plans are adherence to the guidelines.\
Non-Compliance: List the parts that are not adherence to the guidelines and provide corrections or necessary points from the guidelines. If the doctor has plans that go beyond the scope of the guidelines, also point out those plans and explain that they are not in the guidelines.\
Omissions: Highlight any necessary guidelines that the doctor has not addressed in the plans.\
Please ensure that the output is based on the guidelines and do not introduce suggestions outside of the guidance. \
Example: \
Input: \
Query - Medical diagnosis record: \
Reason for admission: fever, cough \
Previous information: 22-year-old male, generally healthy. No medication. Now 1 week of intermittent fever ad 39.5. Lowered with medication until worse in recent days. Hard, dry cough, occasional sputum. Runny nose and sore throat gone.\
Current status: Yt good, but clearly in pain. Temperature in the ear 38.7, painkillers. Rumbling in the right basal lung, otherwise normal sounds. No abnormalities in the heart. Skin clean. Ears and throat clean. No abnormalities on the neck palpation. Skin clean: Sat 96%, heart rate 115 (fever). \
Lab: CRP 25, Leuk 8.9, otherwise normal.\
Thorax x-ray: Right basal infiltrate consistent with pneumonia. Otherwise no abnormal findings. \
Plan: Suitable for pneumonia. Considering the epidemic situation, patient's age and lab results, it could be mycoplasma pneumonia. Start amorion 1g x3 for 10 days and Dximed 100mg x2 alongside. Fever medication Burana + pandol. If not better in 2-3 days or the condition worsens, then a new appointment and a doctor's check-up. \
Diagnosis: J 18.9. Unspecified pneumonia.\
Guidelines:\
Pneumonia refers to an infection of the lung tissue.\
Pneumonia should be suspected if the patient has fever and symptoms of a lower respiratory tract infection (cough and shortness of breath, fever or chest pain) and the illness has affected the general condition, the illness has developed rapidly or the symptoms have worsened again, or if the patient has additional risk factors, such as advanced age or other diseases. In the elderly, the only sign of pneumonia may be confusion.\
The treating physician can identify patients with pneumonia based on the clinical picture and lung auscultation.\
The primary antimicrobial treatment for pneumonia is amoxicillin.\
The duration of antimicrobial treatment is at least 5 days, and the patient must be fever-free for 2–3 days before stopping antimicrobial treatment.\
The majority of pneumonia cases can be treated safely at home.\
The severity of pneumonia can be assessed using the DS-CRB-65 tool.\
For patients over 50 years of age and smokers of all ages who have had pneumonia, a follow-up chest X-ray is recommended after 6–12 weeks.\
Output:\
Pneumonia, in which case at least pneumococcus should be covered (amorion = amoxicillin), suspected of atypical pathogens, in which case doxycycline as a companion."

user_input = "Cause: \
Cough: \
Previous information: \
32-year-old generally healthy woman. General respiratory symptoms for 5-6 days, now the cough is emphasized, mild side pain with coughing fits. No fever, ear or sinus symptoms. \
Current status: \
Good. Breathing free, calm. Heart rate steady, no rhythm. RR 127/80 p 70. Mild mucus rales in the lungs. Ttymp 36.5. Spo2 98%. Rapid CRP 16. \
Plan: \
Impression of viral bronchitis, symptomatic treatment. Speech work: sva 3 days, new connection required. \
Diagnosis: \
J20.9 Unspecified acute bronchitis. \
Guideline: \
Do not use antibiotics for bronchitis without specific reasons. \
Acute bronchitis is a disease limited to the mucous membranes of the bronchi, most often caused by a virus. Antibiotic treatment may slightly shorten the duration of symptoms in acute bronchitis. The cough phase of acute bronchitis can last about three weeks. Recovery is individual. The use of antibiotics increases antibiotic resistance. \
The probability of pneumonia increases if the patient has the following symptoms: \
fever ≥ 37.8 ºC \
rapid breathing (> 16/min) \
tachycardia (over 95/min) \
oxygen saturation < 96% on room air.\
In addition, the following symptoms indicate pneumonia: \
The disease has affected the general condition. \
The symptoms have developed rapidly. \
The symptoms of the respiratory tract infection have worsened again. \
The patient there are additional risk factors (age, other diseases). \
Sudden bronchitis can turn into pneumonia through a mixed infection, especially in the elderly."


prompt = "Olet terveydenhuollon ja kliinisen tutkimuksen asiantuntija, jonka tehtävänä on arvioida, ovatko lääkärin diagnoosit ja suunnitelmat annettujen virallisten ohjeiden mukaisia. Saat kyselynä potilaan kliinisen kertomuksen ja ohjeet.\
Syöte sisältää kaksi osaa:\
1. Kysely - Lääketieteellinen diagnoositietue. Sisältää potilaan perustiedot, tutkimustulokset, lääkärin diagnoosin ja ehdotukset.\
2. Ohjeet. Sisältää ehdotuksia ja vasta-aiheita erilaisiin sairauksiin.\
Tulostesi tulee sisältää seuraavat osat:\
1. Tee yhteenveto potilaan historiasta. \
2. Tee yhteenveto siitä, mitä lääkäri tutki. \
3. Tee yhteenveto lääkärin löydöksistä. \
4. Tee yhteenveto lääkärin diagnoosista. \
5. Tee yhteenveto hoitosuunnitelmasta. \
6. Arvioi, noudattaako kohdassa 2 tiivistämäsi tutkimusstrategia annettuja ohjeita. Miksi tai miksi ei? \
7. Arvioi, onko kohdassa 4 esittämäsi diagnoosi yhteensopiva potilaan historian ja lääkärin löydösten kanssa. \
8. Arvioi, noudattaako kohdassa 5 esittämäsi hoitosuunnitelma annettuja ohjeita. Miksi tai miksi ei? \
9. Korosta tarpeelliset ohjeet, joita lääkäri ei ole käsitellyt hoitosuunnitelmassa. Ohita tämä osa, jos et. \
Varmista, että tulos perustuu ohjeisiin, äläkä esitä ohjeiden ulkopuolisia ehdotuksia. Sano vain, että kysymykseen ei voi vastata, jos et löydä asiaankuuluvaa annettua tietoa tai et tiedä. \
Esimerkki: \
Syöte: \
Kysely - lääketieteellinen diagnoositietue: \
Tulosyy: Yskä. \
Esitiedot: 32-v yleensä terve nainen. 5-6 vrk ajan yleishengitystieoireita, nyt korostuu yskä, yskänpuuskan yhteydessä lievää kylkikipua. Ei kuumetta, korva- tai poskiontelo-oireita. \
Nykytila: Yt hyvä. Hengitys vapaata, rauhallista. Sydämestä tas. rytmi ei sä. RR 127/80 p 70. Keuhkoista lievät limarahinat. Ttymp 36,5. Spo2 98%. Pika-crp 16. \
Suunnitelma: Vaikutelma virusbronkiitista, johon oireenmukainen hoito. Puhetyö: sva 3 vrk, tarv. uusi yhteys. \
Diagnoosi: J20.9 Määrittämätön akuutti keuhkoputkitulehdus \
Ohjeasiakirja: \
Älä ilman erityisiä perusteita käytä antibioottihoitoa keuhkoputkitulehduksen hoidossa. \
Akuutti keuhkoputkitulehdus on keuhkoputkien limakalvoille rajoittuva, useimmiten viruksen aiheuttama sairaus. Antibioottihoito saattaa lyhentää hieman oireiden kestoa akuutissa keuhkoputkitulehduksessa. Akuutin keuhkoputkitulehduksen yskävaihe voi kestää noin kolme viikkoa. Toipuminen on yksilöllistä. Antibioottien käyttö lisää antibioottiresistenssiä. \
Keuhkokuumeen todennäköisyys kasvaa, jos potilaalla on seuraavia oireita: \
kuume ≥ 37,8 ºC. \
tihentynyt hengitys (> 16/min). \
takykardia (yli 95/min). \
happikyllästeisyys < 96 % huoneilmalla. \
Lisäksi keuhkokuumeeseen viittaavat seuraavat oireet: \
Sairaus on vaikuttanut yleiskuntoon. \
Oireet ovat kehittyneet nopeasti. \
Hengitystieinfektion oireet ovat uudestaan vaikeutuneet. \
Potilaalla on lisäriskitekijöitä (ikä, muut sairaudet). \
Äkillinen keuhkoputkitulehdus voi muuttua sekainfektion kautta keuhkokuumeeksi erityisesti iäkkäillä. \
Lähtö: \
1.Potilaalla on ollut muutaman päivän ajan ylähengitysteiden oireita kuten yskää ja lievää kylkikipua. Hänellä ei ole kuumetta tai korvien tai poskionteloiden oireita. \
2.Lääkäri suoritti kliinisen tutkimuksen, johon kuului sydämen ja keuhkojen kuuntelu. Lisäksi lääkäri mittasi verenpaineen, kuumeen, happisaturaation ja tulehdusarvon. \
3.Keuhkoista kuului lievät limarahinat. Tulehdusarvo oli lievästi koholla. Tutkimuksessa ei havaittu muita poikkeavia löydöksiä. \
4.Lääkärin diagnoosi oli viruksen aiheuttama keuhkoputkitulehdus. \
5.Lääkäri suositteli oireenmukaista hoitoa ja kirjoitti sairauslomatodistuksen. \
6.Kysymykseen ei voi vastata, koska annettu hoitosuositus ei käsittele potilaan tutkimista. \
7.Lääkärin asettama diagnoosi sopii esitietoihin ja tutkimuslöydöksiin. \
8.Lääkärin suunnitelma on annetun hoitosuosituksen mukainen, koska keuhkokuumeeseen viittaavia oireita tai löydöksiä ei ollut eikä potilaalle siten määrätty antibioottia."


user_input = "Tulosyy: flunssa. \
Esitiedot: Kyseessä perusterve 35-v nainen, ei sään lääkityksiä. Nyt 2 vko ajan flunssaa. Alkanut kurkkukivulla ja kuumeella. Nyt lähinnä tukkoinen, painetta poskionteloiden alueella. Ei korvakipua, ei enää kuumeilua. \
Nykytila: Yt hyvä, sat 98%, hf rauhallinen. Sydämestä ei ausk poikkeavaa, keuhkoista limaiset, karkeat rahinat l.a. basaalisesti. Siistiytyy yskimisen jälkeen. korvat terveet, hieman alipaineiset. Nielu siisti. Kaulalla palp. reakt suurentuneet imusolmukkeet. RHA siisti, sinusscan -/-. Lämpö 36.5. \
Suunnitelma: Vaikutelma edelleen virustaudista. Ei bakteeri-infektioon viittaavaa. Oirehoitona Nasonex, acriseu. Lisäksi kipulääke tarv. Suositeltu seesamiöljyä kostuttamaa. Uusi yhteys jos vointi heikkenee tai ei helpota. SVA 3pvä. \
Diagnoosi: J06.9. \
Ohjeasiakirja: \
Älä ilman erityisiä perusteita käytä antibioottihoitoa keuhkoputkitulehduksen hoidossa. \
Akuutti keuhkoputkitulehdus on keuhkoputkien limakalvoille rajoittuva, useimmiten viruksen aiheuttama sairaus. Antibioottihoito saattaa lyhentää hieman oireiden kestoa akuutissa keuhkoputkitulehduksessa. Akuutin keuhkoputkitulehduksen yskävaihe voi kestää noin kolme viikkoa. Toipuminen on yksilöllistä. Antibioottien käyttö lisää antibioottiresistenssiä. \
Keuhkokuumeen todennäköisyys kasvaa, jos potilaalla on seuraavia oireita: \
kuume ≥ 37,8 ºC. \
tihentynyt hengitys (> 16/min). \
takykardia (yli 95/min). \
happikyllästeisyys < 96 % huoneilmalla. \
Lisäksi keuhkokuumeeseen viittaavat seuraavat oireet: \
Sairaus on vaikuttanut yleiskuntoon. \
Oireet ovat kehittyneet nopeasti. \
Hengitystieinfektion oireet ovat uudestaan vaikeutuneet. \
Potilaalla on lisäriskitekijöitä (ikä, muut sairaudet). \
Äkillinen keuhkoputkitulehdus voi muuttua sekainfektion kautta keuhkokuumeeksi erityisesti iäkkäillä."





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
