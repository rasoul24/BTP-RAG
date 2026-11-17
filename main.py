import chromadb
import ollama

# 1. Initialisation de la base vectorielle locale (en mémoire pour le test)
client_db = chromadb.Client()
collection = client_db.create_collection(name="normes_btp")

# 2. Documents d'exemple (simulés pour le test)
documents = [
    "Le béton C25/30 est couramment utilisé pour les fondations superficielles et les dalles de bâtiments résidentiels.",
    "Le délai de séchage complet d'une chape traditionnelle en ciment est d'environ 1 semaine par centimètre d'épaisseur.",
    "Le port du casque de chantier, des chaussures de sécurité S3 et du gilet haute visibilité est obligatoire sur toute la zone de gros œuvre.",
    "La norme NF C 15-100 régit les installations électriques basse tension en France.",
]

print("Vectorisation et indexation des documents...")
for i, doc in enumerate(documents):
    # Génération de l'embedding en local via Ollama
    response = ollama.embed(model="bge-m3", input=doc)
    embedding = response["embeddings"][0]

    # Stockage dans ChromaDB
    collection.add(ids=[str(i)], embeddings=[embedding], documents=[doc])

# 3. Question de l'utilisateur
question = "Quel béton utiliser pour les fondations ?"
print(f"\nQuestion : {question}")

# 4. Recherche du passage le plus pertinent (Retrieval)
query_embed = ollama.embed(model="bge-m3", input=question)["embeddings"][0]
results = collection.query(query_embeddings=[query_embed], n_results=1)
contexte = results["documents"][0][0]

print(f"Document retrouvé : {contexte}\n")

# 5. Génération de la réponse finale avec le LLM local (Generation)
prompt = f"""Réponds à la question te basant sur le contexte suivant :
Contexte : {contexte}
Question : {question}"""

reponse = ollama.chat(
    model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}]
)

print("Réponse du RAG :")
print(reponse["message"]["content"])
