from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    return summarizer(text, max_length=130, min_length=40, do_sample=False)


if __name__ == "__main__":
    text = "Tulosyy: Yskä. \
Esitiedot: 32-v yleensä terve nainen. 5-6 vrk ajan yleishengitystieoireita, nyt korostuu yskä, yskänpuuskan yhteydessä lievää kylkikipua. Ei kuumetta, korva- tai poskiontelo-oireita. \
Nykytila: Yt hyvä. Hengitys vapaata, rauhallista. Sydämestä tas. rytmi ei sä. RR 127/80 p 70. Keuhkoista lievät limarahinat. Ttymp 36,5. Spo2 98%. Pika-crp 16. \
Suunnitelma: Vaikutelma virusbronkiitista, johon oireenmukainen hoito. Puhetyö: sva 3 vrk, tarv. uusi yhteys. \
Diagnoosi: J20.9 Määrittämätön akuutti keuhkoputkitulehdus."

    print("<<< Summary text:", summarize(text))
