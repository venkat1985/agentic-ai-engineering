documents=[
    {
        "id": "doc-1",
        "text": "Visa payments are failing due to issuer timeouts.",
        "metadata": {
            "payment_network": "Visa",
            "environment": "Production",
            "region": "US"
        }
    },
{
        "id": "doc-2",
        "text": "Mastercard payments are experiencing issuer declines.",
        "metadata": {
            "payment_network": "Mastercard",
            "environment": "Production",
            "region": "US"
        }
    },
{
        "id": "doc-3",
        "text": "Visa payments failed during staging testing.",
        "metadata": {
            "payment_network": "Visa",
            "environment": "staging",
            "region": "US"
        }
    },
    {
        "id": "doc-4",
        "text": "Visa payments are experiencing acquirer timeouts.",
        "metadata": {
            "payment_network": "Visa",
            "environment": "Production",
            "region": "EU"
        }
    }
]

def filter_documents(documents, filters):
    filtered=[]

    for doc in documents:
        metadata=doc["metadata"]
        matches=all(metadata.get(key) == value for key, value in filters.items())
        if matches:
            filtered.append(doc)
    return filtered
if __name__ == "__main__":
    filters={
        "payment_network":"Visa",
        "environment":"Production",
        "region": "US"
    }
    results=filter_documents(documents,filters)

    print("\n====FILTERED DOCUMENTS ========")
    for doc in results:
        print("ID:",doc["id"])
        print("Text:",doc["text"])
        print("Metadata:",doc["metadata"])