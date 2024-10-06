import json
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi import  APIRouter
from app.analysis import  classify_single_instance
from app.models import DatasetInfo
from app.utils import  count_languages, count_null_lang, count_null_toxicity, lang_distribution, toxicity_distribution
from datasets import load_dataset
from starlette.responses import StreamingResponse
from fastapi.responses import StreamingResponse
import time

router = APIRouter()

df = pd.DataFrame()
processed_df = pd.DataFrame()
dataset_name = ""
current_distribution = {}
streaming = True



@router.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    global df, dataset_name
    try:
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "csv":
            df = pd.read_csv(file.file)
        
        elif file_extension == "json":
            df = pd.read_json(file.file)
        
        elif file_extension == "jsonl":
            df = pd.read_json(file.file, lines=True)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV, JSON, or JSONL file.")
        
        dataset_name = file.filename

        return {"message": "Dataset uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading dataset: {e}")
    

@router.get("/use-default-dataset/")
async def use_default_dataset():
    global df, dataset_name
    try:
        dataset = load_dataset('OpenAssistant/oasst2')
        
        df = dataset['train'].to_pandas()
        dataset_name = "OpenAssistant/oasst2"
        
        return {"message": "Default dataset loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load the default dataset: {str(e)}")


@router.get("/dataset-info/")
async def get_dataset_info():
    global df, dataset_name
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset is loaded. Please upload a dataset or use the default dataset.")
    
    try:
        dataset_info = {
            "name": dataset_name,
            "num_instances": len(df),
            "num_attributes": len(df.columns),
            "lang_count": count_languages(df)  
        }
        return DatasetInfo(**dataset_info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving dataset info: {e}")
    

@router.get("/language-distribution/")
async def get_language_distribution():
    global df  
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset is loaded. Please upload a dataset or use the default dataset.")
    
    try:
        distribution = lang_distribution(df)
        return {"language_distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving language distribution: {e}")


@router.get("/lang-null-count/")
async def get_lang_null_count():
    global df
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    
    try:
        null_count_info = count_null_lang(df)  
        return {
            "null_count": int(null_count_info["count"]),  
            "percentage": float(null_count_info["percentage"]) 
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error counting null values: {e}")


@router.get("/toxicity-null-count/")
async def get_toxicity_null_count():
    global df
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    
    try:
        null_count_info = count_null_toxicity(df)  
        return {
            "null_count": int(null_count_info["count"]),  
            "percentage": float(null_count_info["percentage"])  
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error counting null values: {e}")

@router.get("/toxicity-distribution/")
async def get_toxicity_distribution():
    global df
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    
    try:
        distribution = toxicity_distribution(df)
        return {"toxicity_distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating toxicity distribution: {e}")


@router.get("/classify-intents/")
async def classify_intents_stream():
    async def intent_generator():
        global df, streaming 
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is not uploaded or is empty")

        prompter_df = df[df['role'] == 'prompter'].copy()
        
        texts = prompter_df['text']

        start = time.time()
        processed_count = 0

        intent_distribution = {
            "Summarization": 0,
            "Translation": 0,
            "Paraphrasing": 0,
            "Role-play": 0,
            "Miscellaneous": 0
        }


        for text in texts:  
            if not streaming:  
                break


            intent = await classify_single_instance(text)
            processed_count += 1

            if intent in intent_distribution:
                intent_distribution[intent] += 1
            else:
                intent_distribution["Miscellaneous"] += 1  

            print(f"Processed: {processed_count} / {len(texts)}")  
            
            yield f"data: {json.dumps({'text': text, 'intent': intent, 'processed_count': processed_count, 'total': len(texts), 'elapsed_time': time.time() - start, 'intent_distribution': intent_distribution})}\n\n"

    
    headers = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream"
    }

    return StreamingResponse(intent_generator(), media_type="text/event-stream", headers=headers)


@router.post("/stop-stream/")
async def stop_stream():
    global streaming
    streaming = False  
    print("Stream stopped")
    return {"message": "Stream stopped"}

