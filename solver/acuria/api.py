"""API HTTP do laboratório de modelos."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from acuria.dataset import OriginError
from acuria.schemas import ExplainRequest, LogsResponse, OptimizeRequest
from acuria.service import BusyError, Lab, MissingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

lab = Lab()
app = FastAPI(
    title="Acuria",
    summary="Otimização de modelos de diagnóstico por algoritmo genético",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def count_requests(request, call_next):
    lab.monitor.requests += 1
    return await call_next(request)


@app.exception_handler(BusyError)
async def busy_handler(_request, exc: BusyError):
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(MissingError)
async def missing_handler(_request, exc: MissingError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(OriginError)
async def origin_handler(_request, exc: OriginError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_handler(_request, exc: Exception):
    if isinstance(exc, (HTTPException, RequestValidationError)):
        raise exc
    lab.monitor.failures += 1
    logging.getLogger("acuria").exception("falha interna")
    return JSONResponse(status_code=500, content={"detail": "Falha interna ao processar o modelo."})


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "acuria"}


@app.get("/api/overview")
def overview(source: str = "data"):
    return lab.overview(source)


@app.post("/api/optimize")
def optimize(request: OptimizeRequest):
    return lab.optimize(request)


@app.get("/api/runs/{run_id}/cases")
def cases(run_id: str):
    return lab.cases(run_id)


@app.post("/api/runs/{run_id}/explain")
def explain(run_id: str, request: ExplainRequest):
    return lab.explain(run_id, request.case_index)


@app.post("/api/runs/{run_id}/report")
def report(run_id: str):
    return lab.report(run_id)


@app.get("/api/metrics")
def metrics():
    return lab.metrics()


@app.get("/api/logs")
def logs():
    return LogsResponse(events=lab.logs())
