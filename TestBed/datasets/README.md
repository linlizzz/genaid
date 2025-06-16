## Guidelines:

`Käypä_hoito.jsonl  ## Recommendations`

`Vältä_viisaasti.jsonl  ## Avoid Wisely Recommendations`

`Current_Care_Guidelines.jsonl  ## translated (full?) Käypä hoito guidelines`

`Current_Care_Summary.jsonl  ## contains summaries of selected Käypä hoito guidelines translated to English`

Sequential chunks from the page's content, each ~ 200 words. 

Created by direct segmentation — no consideration was given to semantic similarity between sentences. 

Table information on the page was included in the chunks, but image content was excluded. 

Each chunk begins with the corresponding headings from different levels on that page.

##### Format: 
    {
      "guideline_id": "hoi_04010",
      "title": "Kohonnut verenpaine",
      "keywords": ['Sisätaudit', 'Kardiologia', 'Kliininen farmakologia', …],
      "chunks": [
      {
        "chunk_id": "hoi_04010_chunk_01",
        "page_content": "..."
      },
      {
        "chunk_id": "hoi_04010_chunk_02",
        "page_content": "..."
      },
      ...
      ]
    }

`Käypä_hoito_flat.jsonl`

`Vältä_viisaasti_flat.jsonl`

`Current_Care_Guidelines_flat.jsonl`

`Current_Care_Summary_flat.jsonl`

Contain the original, unchunked content of each page.

##### Format: 
    {
      "guideline_id": "hoi_04010",
      "title": "Kohonnut verenpaine",
      "keywords": ['Sisätaudit', 'Kardiologia', 'Kliininen farmakologia', …],
      "page_content": "the complete content of each page as-is, without splitting it into chunks."
    }

## Clinical notes:

`clnical_notes.jsonl`

##### Format: 
    {
      "note_id": "note_001",
      "text": "...patient story text...",
      "linked_guideline_ids": ["hoi_04010", "hoi_05030"]
    }

## Annotation:

`annotation.jsonl`

