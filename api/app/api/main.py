from fastapi import FastAPI
from app.schemas.model_name_schema import ModelName
from app.services.ai_service import load_models

app = FastAPI()

models = load_models()

@app.post("/guess")
async def guess(
    text: str,
    model: ModelName = ModelName.naive_bayes
):
    ai_model = models[model]
    prediction = ai_model(text)

    total_sum = sum(prediction.values())
    return {
        "prediction": prediction,
        "model": model,
        "total_sum": total_sum,
    }
