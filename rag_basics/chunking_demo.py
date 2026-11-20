from langchain_text_splitters import RecursiveCharacterTextSplitter


document = """
Payment authorization requests are sent to the issuer.
The issuer can approve or decline the transaction.
Declines may happen because of insufficient funds.

Payment processing can experience timeouts.
A timeout occurs when a downstream service does not
respond within the configured response timeout.

Circuit breakers protect applications from repeated
downstream failures. When the failure threshold is
reached, the circuit breaker opens and rejects new
requests until recovery is attempted. 
Most of the times, it sends the traffic to other acquirers to process the Payments.
"""
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40)

chunks=splitter.split_text(document)

print("Total chunks:",len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\n------chunk{i+1} ----")
    print(chunk)
    print("Characters:",len(chunk))

