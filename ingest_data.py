import requests

sample_docs = [
    {
        "text": "PM Kisan Samman Nidhi is a government scheme that provides Rs 6000 per year to farmers in three installments of Rs 2000 each. Farmers with less than 2 hectares of land are eligible.",
        "source": "pm_kisan.txt"
    },
    {
        "text": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana provides health insurance cover of Rs 5 lakh per family per year for secondary and tertiary care hospitalization to over 10 crore poor families.",
        "source": "ayushman_bharat.txt"
    },
    {
        "text": "Smart Cities Mission aims to develop 100 cities across India making them citizen friendly and sustainable. Focus areas include infrastructure, technology, and governance.",
        "source": "smart_cities.txt"
    },
    {
        "text": "Swachh Bharat Abhiyan is a nationwide cleanliness campaign launched in 2014. It aims to achieve an open defecation free India and improve solid waste management.",
        "source": "swachh_bharat.txt"
    },
    {
        "text": "Pradhan Mantri Awas Yojana provides affordable housing to urban and rural poor. The scheme provides financial assistance for construction or enhancement of houses.",
        "source": "pmay.txt"
    },
    {
        "text": "National Digital Health Mission aims to develop the necessary backbone for digital health services in India including health IDs, digital health records, and telemedicine.",
        "source": "ndhm.txt"
    },
]

texts = [d["text"] for d in sample_docs]
metadatas = [{"source": d["source"]} for d in sample_docs]

response = requests.post(
    "http://localhost:8000/ingest",
    json={"texts": texts, "metadatas": metadatas}
)
print("Ingest result:", response.json())