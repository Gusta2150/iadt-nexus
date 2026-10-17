from fastapi.testclient import TestClient

from acuria.api import app


def test_saude_visao_geral_e_leitura():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        overview = client.get("/api/overview")
        assert overview.status_code == 200
        body = overview.json()
        assert body["samples"] == 569
        assert len(body["baselines"]) == 4
        assert body["split"]["train"] + body["split"]["validation"] + body["split"]["test"] == 569

        optimized = client.post(
            "/api/optimize",
            json={
                "model": "arvore",
                "population_size": 6,
                "generations": 2,
                "seed": 4,
                "experiments": False,
            },
        )
        assert optimized.status_code == 200
        payload = optimized.json()
        original = next(item for item in payload["runs"] if item["label"] == "Modelo original")
        evolved = next(item for item in payload["runs"] if item["label"] == "Algoritmo genético")
        assert evolved["validation"]["fitness"] + 1e-9 >= original["validation"]["fitness"]
        assert payload["importance"]

        cases = client.get(f"/api/runs/{payload['run_id']}/cases")
        assert cases.status_code == 200
        assert len(cases.json()["cases"]) == body["split"]["test"]

        explained = client.post(
            f"/api/runs/{payload['run_id']}/explain",
            json={"case_index": 0},
        )
        assert explained.status_code == 200
        reading = explained.json()
        assert reading["source"] == "local"
        assert reading["quality"]["score"] >= 85
        assert "não substitui" in reading["text"].casefold()

        report = client.post(f"/api/runs/{payload['run_id']}/report")
        assert report.status_code == 200
        assert report.json()["quality"]["score"] >= 85

        missing = client.post("/api/runs/nao-existe/explain", json={"case_index": 0})
        assert missing.status_code == 404

        metrics = client.get("/api/metrics")
        assert metrics.json()["optimizations"] >= 1
        logs = client.get("/api/logs")
        assert logs.json()["events"]
