import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from engine import OpenAIvLLMEngine, vLLMEngine
from utils import JobInput

vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)

app = FastAPI()


async def vllm_response_transformer(response_generator):
    async for chunk in response_generator:
        print(chunk)
        yield chunk

    pass


@app.api_route("/openai/{full_path:path}", methods=["GET", "POST"])
async def openai_proxy(full_path: str, request: Request):
    try:
        # Attempt to parse request body as JSON
        body = await request.json()
    except json.JSONDecodeError as e:
        print("Error parsing request body", e)
        # If body is not JSON or empty, use an empty dictionary
        body = {}

    is_stream = body.get("stream", False)

    # Construct JobInput object expected by the engine
    # The full_path from the URL becomes the openai_route
    job_input = JobInput(
        {"openai_route": f"/{full_path}", "openai_input": body, "stream": is_stream}
    )

    response_generator = openai_engine.generate(job_input)

    if is_stream:
        # Call the generate method of the OpenAIvLLMEngine
        response_generator = vllm_response_transformer(response_generator)

        # Return a StreamingResponse from the generator
        # Assuming the generator yields data in a format suitable for text/event-stream
        return StreamingResponse(response_generator, media_type="text/event-stream")
    else:
        raise HTTPException(status_code=500, detail="Unimplemented")


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
