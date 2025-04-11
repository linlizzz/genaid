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

#  Kommentti: Kuvattu toiminta on oleellisesti hoitosuositusten mukaista.


if __name__ == '__main__':

    # print('Chat with documents (type "exit" to quit)')
    # chat_history = []
    # while True:
        # user_input = input("\n\n>>> Input: ")
        # if query.lower() == 'exit':
            # break
        # elif len(query) == 0:
            # continue
    query = user_input
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    
    pipe = pipeline("text-generation", model="LumiOpen/Poro-34B-chat", torch_dtype=torch.bfloat16, device="cuda", max_new_tokens=1000)
    
    response = pipe(messages)
        #response = client.generate(model=model, prompt=user_input)
    print("----Poro-34B-chat----\n\n")
    print("<<< Response:", response)
