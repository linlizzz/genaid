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


if __name__ == '__main__':

    # print('Chat with documents (type "exit" to quit)')
    # chat_history = []
    # while True:
        # user_input = input("\n\n>>> Input: ")
        # if query.lower() == 'exit':
            # break
        # elif len(query) == 0:
            # continue
    # models = ["LumiOpen/Poro-34B-chat", "BioMistral/BioMistral-7B", "utter-project/EuroLLM-9B-Instruct"]
    models = ["utter-project/EuroLLM-9B-Instruct"]

    for model in models:
        pipe = pipeline("text-generation", model="utter-project/EuroLLM-9B-Instruct", torch_dtype=torch.bfloat16, device="cuda", max_new_tokens=1000)
        
        query_summary = user_input
        messages_summary = [
            {"role": "system", "content": prompt_summary},
            {"role": "user", "content": query_summary}
        ]
        response_summary_all = pipe(messages_summary)
            #response = client.generate(model=model, prompt=user_input)
        response_summary = response_summary_all[0]['generated_text'][-1]['content']
        print("----", model,"----\n\n")
        print("<<< Response:", response_summary)
        print("\n\n")


        query_evaluation = response_summary + "\n" + guideline
        messages_evaluation = [
            {"role": "system", "content": prompt_evaluation},
            {"role": "user", "content": query_evaluation}
        ]
        response_evaluation_all = pipe(messages_evaluation)
        response_evaluation = response_evaluation_all[0]['generated_text'][-1]['content']
        print("<<< Response:", response_evaluation)

