# FastAPI app providing /chat endpoint for NPC dialogue

import json
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# --- Load Personas ---
with open("personas.json", encoding="utf-8") as f:
    PERSONAS = json.load(f)

# --- Initialize Retrieval ---
index  = faiss.read_index("data/lore.index")
with open("data/lore_meta.json", encoding="utf-8") as f:
    META = json.load(f)
retriever = SentenceTransformer("all-mpnet-base-v2")

def retrieve(query: str, k: int = 4):
    q_vec = retriever.encode(query).reshape(1, -1)
    distances, indices = index.search(q_vec, k)
    return [META[idx]["text"] for idx in indices[0]]

# --- Initialize Generation Model ---
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", low_cpu_mem_usage=True)

def build_prompt(npc_id: str, player_input: str, lore_snips: List[str]) -> str:
    # 1) Load this NPC’s persona fields from personas.json
    p = PERSONAS[npc_id]

    # 2) Build a one-sentence “ROLEPLAY” instruction (name, role, location, tone, quirk)
    instr_parts = [f"ROLEPLAY: You are {p['name']}"]
    if p.get("role"):
        instr_parts.append(f"the {p['role']}")
    if p.get("location"):
        instr_parts.append(f"based in {p['location']}")
    if p.get("tone"):
        instr_parts.append(f"and speak in a {p['tone']} tone")
    if p.get("quirk"):
        instr_parts.append(f"Quirk: {p['quirk']}")
    persona_instr = " ".join(instr_parts) + "."

    # 3) Number all the lore snippets
    lore_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(lore_snips))

    # 4) Add a strict formatting directive:
    formatting_instr = (
        "INSTRUCTIONS: Using only your own in-character voice, "
        "answer the user in exactly 2–3 sentences. "
        "Do not restate your persona, do not add any out-of-character commentary, "
        "and explicitly reference details from the lore snippets provided."
    )

    # 5) Decide how the model knows when to speak:
    speak_cue = p['name'].split()[0]

    return (
        f"{persona_instr}\n\n"
        f"Relevant lore (use these to ground your answer; reference specific details if possible):\n"
        f"{lore_text}\n\n"
        f"{formatting_instr}\n\n"
        f"User: {player_input}\n"
        f"{speak_cue}:"
    )

def simple_safety_filter(text: str) -> str:
    deny = {"damn", "hell", "bloody"}
    lower = text.lower()
    if any(bad in lower for bad in deny):
        return "[…]"
    return text

def generate_reply(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # get only the newly generated tokens, not the prompt
    gen_tokens = out[0][ inputs["input_ids"].shape[-1]: ]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# --- FastAPI Endpoint ---
app = FastAPI()

class ChatRequest(BaseModel):
    npc_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.npc_id not in PERSONAS:
        return {"error": f"Unknown npc_id: {req.npc_id}"}
    lore    = retrieve(req.message)
    prompt  = build_prompt(req.npc_id, req.message, lore)
    reply   = generate_reply(prompt)
    return {"reply": reply}

# To run:
#   uvicorn rag_server:app --reload --host 0.0.0.0 --port 8000
