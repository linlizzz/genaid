### Raw Data
#### ./clinical_notes_data/
.docx files, 8 patient stories in Finnish.

#### ./Current Care Guidelines/
.xml files, raw page contents in English.

#### ./Current Care Summary/
.xml files, raw page contents in English.

#### ./Käypä hoito/ 
.xml files, raw page contents (current treatment) in Finnish.

#### ./Vältä viisaasti/
.xml files, raw page contents (avoid wisely) in Finnish.


### Preproccessed Data
#### ./Current_Care_Guidelines_metadata
.csv files, metadata extracted in English. Column: page_content, keywords, title

#### ./Current_Care_Summary_metadata
.csv files, metadata extracted in English. Column: page_content, keywords, title

#### ./Käypä_hoito_metadata/
.csv files, metadata extracted (current treatment) in Finnish. Column: page_content, keywords, title

#### ./Vältä_viisaasti_metadata/
.csv files, metadata extracted (current treatment) in Finnish. Column: page_content, keywords, title

#### ./samples/
Potilastarinoita.csv. Column: Tarina (story), Arvioitava hoitosuositus (Evaluable Treatment Recommendation - Guidelines), Mallivastaus (Sample Answer).

Arvioitava hoitosuositus.csv, Column: Arvioitava hoitosuositus (All guidelines which might use in the given patient's stories.)

guideline.txt/guideline_eng.txt. All guidelines which might use in the given patient's stories in Finnish/English.