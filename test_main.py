from fastapi.testclient import TestClient
import json 


from main import app


client = TestClient(app)


def test_readyness():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'hello':'world'}


def test_noAC():
    data = {"text":"Hola Mundo"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["AssetClass"] == ["None"]


def test_checkPos():
    data = {"text":"Positive results for US equity"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["Pos"] > resp["Neg"] 


def test_checkNeg():
    data = {"text":"Negative results for US equity"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["Neg"] > resp["Pos"] 


def test_checkDouble():
    data = {"text":"Negative results for US equity and Emerging Markets"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["doc_type"] == ['2','2'] 
    assert len(resp["AssetClass"]) == 2 
    assert resp["AssetClass"] == ['us-equity', 'emer-equity']

    