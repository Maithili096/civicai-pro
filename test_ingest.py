import requests

r = requests.post("http://127.0.0.1:8002/ingest", json={
    "texts": [
        "PM Kisan provides Rs 6000 per year to farmers.",
        "Ayushman Bharat provides Rs 5 lakh health insurance.",
        "Swachh Bharat cleanliness campaign launched 2014."
    ],
    "metadatas": [
        {"source": "pm_kisan.txt"},
        {"source": "ayushman.txt"},
        {"source": "swachh.txt"}
    ]
})
print(r.json())

r2 = requests.post("http://127.0.0.1:8002/query", json={"query": "Tell me about PM Kisan"})
print(r2.json())
