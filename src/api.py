from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import RootModel
from typing import Dict, List
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

app = FastAPI()

MODEL_PATH = "pii_bert"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

class PIIRequest(RootModel[Dict[str, List[str]]]):
    pass

@app.post("/detect-pii")
def detect_pii(request: PIIRequest) -> Dict[str, List[dict]]:
    data = request.root
    result = {}
    for col, values in data.items():
        if not values:
            result[col] = [{}, {"count": 0}]
            continue
        inputs = tokenizer(values, truncation=True, padding=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).numpy()
        count_true = sum(bool(pred) for pred in preds)
        word_map = {word: bool(pred) for word, pred in zip(values, preds)}
        result[col] = [word_map, {"count": count_true}]
    return result

@app.post("/csv-to-parquet")
async def csv_to_parquet(file: UploadFile = File(...)):
    temp_csv_path = f"temp_{file.filename}"
    with open(temp_csv_path, "wb") as f:
        f.write(await file.read())
    df = pd.read_csv(temp_csv_path)
    table = pa.Table.from_pandas(df)
    parquet_path = temp_csv_path.replace(".csv", ".parquet")
    pq.write_table(table, parquet_path)
    os.remove(temp_csv_path)
    def iterfile():
        with open(parquet_path, "rb") as f:
            yield from f
        os.remove(parquet_path)
    return StreamingResponse(iterfile(), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={os.path.basename(parquet_path)}"})

@app.post("/parquet-to-csv")
async def parquet_to_csv(file: UploadFile = File(...)):
    temp_parquet_path = f"temp_{file.filename}"
    with open(temp_parquet_path, "wb") as f:
        f.write(await file.read())
    table = pq.read_table(temp_parquet_path)
    df = table.to_pandas()
    csv_path = temp_parquet_path.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)
    os.remove(temp_parquet_path)
    def iterfile():
        with open(csv_path, "rb") as f:
            yield from f
        os.remove(csv_path)
    return StreamingResponse(iterfile(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={os.path.basename(csv_path)}"})