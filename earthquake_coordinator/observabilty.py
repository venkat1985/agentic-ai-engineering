import time

def invoke_with_metrics(model,prompt,agent_name):
    start=time.perf_counter()

    try:
        response=model.invoke(prompt)
        duration=time.perf_counter()-start

        usage=response.usage_metadata or {}

        input_tokens=usage.get("input_tokens",0)
        output_tokens=usage.get("output_tokens",0)
        total_tokens=usage.get("total_tokens",0)

        print(f"\n======{agent_name.upper()} METRICS========")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")

        return response
    except Exception as e:
        duration= time.perf_counter() - start
        print(f"\n{agent_name} failed after duration {duration:.2f}s")
        print(f"Error: {e}")
    raise



