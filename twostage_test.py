from finnish_llm_test import finnish_llm
from biomistral_test import english_llm


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

prompt_evaluation = "Olet terveydenhuollon ja kliinisen tutkimuksen asiantuntija, jonka vastuulla on arvioida, ovatko lääkärin diagnoosi ja suunnitelmat annettujen ohjeiden mukaisia. Saat yhteenvedon potilaan kliinisistä tiedoista ja ohjeista.\
Syöte sisältää kaksi osaa: \
1. Yhteenveto potilaan historiasta ja yleiskunnosta, mitä lääkäri tutki, lääkärin tutkimustulokset, lääkärin diagnoosi ja hoitosuunnitelma. \
2. Ohjeistus, jota sinun tulisi käyttää referenssinä. \
Tuloste sisältää kolme osaa: \
1. Arvioi, noudattaako tutkimusstrategia annettuja ohjeita. Miksi tai miksi ei? \
2. Arvioi, onko diagnoosi yhteensopiva potilaan historian ja lääkärin havaintojen kanssa. \
3. Arvioi, noudattaako hoitosuunnitelma annettuja ohjeita. Miksi tai miksi ei? \
Varmista, että tuloste perustuu ohjeeseen, äläkä sisällytä syötteen ulkopuolista sisältöä. Sano vain, että kysymykseen ei voida vastata, jos et löydä asiaankuuluvaa tietoa tai et tiedä. Tee tulosteesta ytimekäs mutta kattava. \
Esimerkki: \
Syöte: \
Yhteenveto: 1. Potilaalla on ollut kuumetta ja kurkkukipua 24 tuntia, johon liittyy kipua niellessä. \
2. Lääkäri arvioi potilaan yleisen tilan, kuunteli sydäntä ja keuhkoja, tutki kurkun, korvat ja kaulan sekä mittasi C-reaktiivisen proteiinin. \
3. Nielurisat olivat selvästi punaiset ja turvonneet. Kaulassa havaittiin suurentuneita imusolmukkeita. CRP oli 120, mikä on selvästi koholla. \
4. Lääkäri diagnosoi potilaalla nielutulehduksen. \
5. Lääkäri lähetti potilaan korva-, nenä- ja kurkkutautien ensiapuun. \
Ohje: Älä määritä CRP:tä, kun epäilet viruksen tai streptokokki A:n (StrA) aiheuttamaa nielutulehdusta. \
Älä ota nielunäytettä lieväoireiselta (Centor-pisteet 0-2) nielukipupotilaalta. Nielukivun Centor-pisteytys: Yskän puuttuminen, 1 piste; Leukakulman alaisten imusolmukkeiden aristus ja turvotus, 1 piste; Nielurisojen turvotus tai peitteet, 1 piste; Esitietoihin perustuva tai mitattu kuume yli 38 °C, 1 piste. \
Älä käytä muita mikrobilääkkeitä kuin V-penisilliiniä akuuttiin nielutulehdukseen, jos potilaalla ei ole vasta-aihetta penisilliinille. Toistuvan infektion suositeltava lääkitys on ensimmäisen polven kefalosporiini. \
Tulos: \
1. Aktiviteetti ei noudata suositusta - CRP on tarpeeton kurkkukipupotilaan diagnosoinnissa, koska CRP:tä ei yleensä pitäisi testata, kun epäillään nielutulehdusta. \
2. Nielutulehduksen diagnoosi vastaa alustavia tietoja ja tutkimustuloksia. \
3. Kysymykseen ei voida vastata annettujen tietojen perusteella."


user_input = "Esitiedot: 16-v poika, jolla 2-3 vrk ajan nuhaa, lievää yskää, kurkkukipua. Äiti epäilee angiinaa. \
Nykytila: Äidin kanssa vo:lla. Yt hyvä, hengitys vapaata rauhallista. Sydämestä ja keuhkoista ei kuulu poikkeavaa. Kaulalta ei palpoidu poikkeavaa. Nielu diffuusisti punoittaa, risat kenties hieman turvonneet. \
Suunnitelma: Streptokokki-infektion poissulkuun nieluviljely, tarkistavat vastauksen itse OmaKannasta. Muuten oireenmukainen hoito. \
Diagnoosi: J06.9 Määrittämätön akuutti ylähengitystieinfektio."

guideline = "Ohje: Älä määritä CRP:tä, kun epäilet viruksen tai streptokokki A:n (StrA) aiheuttamaa nielutulehdusta. \
Älä ota nielunäytettä lieväoireiselta (Centor-pisteet 0-2) nielukipupotilaalta. Nielukivun Centor-pisteytys: Yskän puuttuminen, 1 piste; Leukakulman alaisten imusolmukkeiden aristus ja turvotus, 1 piste; Nielurisojen turvotus tai peitteet, 1 piste; Esitietoihin perustuva tai mitattu kuume yli 38 °C, 1 piste. \
Älä käytä muita mikrobilääkkeitä kuin V-penisilliiniä akuuttiin nielutulehdukseen, jos potilaalla ei ole vasta-aihetta penisilliinille. Toistuvan infektion suositeltava lääkitys on ensimmäisen polven kefalosporiini."


# models = ["LumiOpen/Poro-34B-chat", "BioMistral/BioMistral-7B", "utter-project/EuroLLM-9B-Instruct"]

# Stage 1: Summarise the patient's history and current condition
# The output could include:
# 1.	Summarise the patient history.
# 2.	Summarise what the doctor examined.
# 3.	Summarise the doctor’s findings
# 4.	Summarise the doctor’s diagnosis.
# 5.	Summarise the treatment plan.

response_summary = finnish_llm(model="LumiOpen/Poro-34B-chat", user_input=user_input, prompt=prompt_summary)

eval_input_eng = finnish_llm(model="LumiOpen/Poro-34B-chat", user_input="Yhteenveto: " + response_summary + "\n Ohje: " + guideline, prompt="translate the whole text into english version")
# Stage 2: Evaluate the doctor's diagnosis and treatment plan based on the guidelines
# The output could include:
# 6.	Evaluate if the examination strategy the doctor used (in 2) follows the given guidelines. Why or why not?
# 7.	Evaluate if the diagnosis the doctor made (in 4) is compatible with the patient history and doctor’s findings.
# 8.	Evaluate if the treatment plan (in 5) follows the given guidelines. Why or why not?

response_evaluation = english_llm(user_input=eval_input_eng, model="BioMistral/BioMistral-7B")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("LumiOpen/Poro-34B-chat")
# model = AutoModelForCausalLM.from_pretrained("LumiOpen/Poro-34B-chat")
'''
fin_model = ["LumiOpen/Poro-34B-chat", "utter-project/EuroLLM-9B-Instruct"]
eng_model = ["BioMistral/BioMistral-7B"]
model_summary = "LumiOpen/poro-34b-chat"
model_evaluation = "BioMistral/BioMistral-7B"
if __name__ == "__main__":
    if model_summary in fin_model:
        response = finish_llm(model=model_summary, user_input=user_input, prompt=prompt_summary)
    else:
        fin_to_eng = finish_llm(model=model_summary, user_input=user_input, prompt="translate to english")
        response = finish_llm(model=model_evaluation, user_input=fin_to_eng, prompt=prompt_evaluation)
'''