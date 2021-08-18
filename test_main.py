from fastapi.testclient import TestClient

from main import app

import json 
client = TestClient(app)


def test_readyness():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'hello':'worldhgjnsa'}

def test_noAC():
    data = {"text":"Hola Mundo"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["AssetClass"] == ["None"]