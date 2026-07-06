from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from . import chatbot_api, module2_api, module3_api, module4_api, module6_api, module7_api, winner_api


app = FastAPI(
    title="F1 Capstone Project",
    description="Module implementation using fastapi",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(winner_api.router)
app.include_router(module2_api.router)
app.include_router(module3_api.router)
app.include_router(module4_api.router)
app.include_router(chatbot_api.router)
app.include_router(module6_api.router)
app.include_router(module7_api.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request body or query parameters.",
            "errors": exc.errors(),
            "tip": "Open /docs and use the example JSON shown for this endpoint.",
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "message": "Unexpected backend error.",
            "error": str(exc),
        },
    )


@app.get("/")
def health_check() -> dict:
    return {
        "status": "ok",
        "modules": [
            "module1 winner prediction",
            "module2 lap time prediction",
            "module3 driver performance classification",
            "module4 pit stop strategy",
            "module5 chatbot",
            "module6 race summary",
            "module7 explainable AI",
        ],
    }
