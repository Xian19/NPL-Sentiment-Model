from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy
from pydantic import BaseModel

app = FastAPI(tags=['sentence'])

# Load your custom spaCy model
nlp_ner = spacy.load("model-best")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # Add the URL where your React app is hosted
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    sentence: str

@app.post("/analyze_text")
def get_text_entities(sentence_input: Input):
    document = nlp_ner(sentence_input.sentence)
    entities = []

    for ent in document.ents:
        entity_info = {
            "Text": ent.text,
            "Start": ent.start_char,
            "End": ent.end_char,
            "Label": ent.label_
        }
        entities.append(entity_info)

    return {"entities": entities}
