import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from FlagEmbedding import FlagAutoModel

app = FastAPI(title="M3 Embedding Service")

# 모델 전역 변수
global_model = None


class TextRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = None
    return_dense: Optional[bool] = True
    return_sparse: Optional[bool] = False
    return_colbert_vecs: Optional[bool] = False
    batch_size: Optional[int] = 256
    max_length: Optional[int] = 512


class TextPairRequest(BaseModel):
    sentence_pairs: List[List[str]]  # [[query, passage], ...]
    weights_for_different_modes: Optional[List[float]] = None
    batch_size: Optional[int] = 256
    max_query_length: Optional[int] = 512
    max_passage_length: Optional[int] = 512


class EmbeddingResponse(BaseModel):
    dense_vecs: Optional[List[List[float]]] = None
    lexical_weights: Optional[List[Dict[str, float]]] = None
    colbert_vecs: Optional[List[List[List[float]]]] = None


class ScoreResponse(BaseModel):
    colbert: Optional[List[float]] = None
    sparse: Optional[List[float]] = None
    dense: Optional[List[float]] = None
    sparse_dense: Optional[List[float]] = None
    colbert_sparse_dense: Optional[List[float]] = None


@app.on_event("startup")
async def load_model():
    """서버 시작시 모델 로드"""
    global global_model
    try:
        global_model = FlagAutoModel.from_finetuned(
            model_name_or_path="BAAI/bge-m3",
            normalize_embeddings=True,
            use_fp16=True,
            devices="cuda:0",
            model_class="encoder-only-m3",
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: TextRequest):
    """텍스트 임베딩 생성"""
    if global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 임베딩 생성
        outputs = global_model.encode(
            sentences=request.texts,
            batch_size=request.batch_size,
            max_length=request.max_length,
            return_dense=request.return_dense,
            return_sparse=request.return_sparse,
            return_colbert_vecs=request.return_colbert_vecs,
            instruction=request.instruction,
        )

        # numpy arrays를 리스트로 변환
        response = {}
        if "dense_vecs" in outputs and outputs["dense_vecs"] is not None:
            response["dense_vecs"] = (
                outputs["dense_vecs"].tolist()
                if len(outputs["dense_vecs"].shape) == 2
                else [outputs["dense_vecs"].tolist()]
            )

        if "lexical_weights" in outputs and outputs["lexical_weights"] is not None:
            response["lexical_weights"] = (
                outputs["lexical_weights"]
                if isinstance(outputs["lexical_weights"], list)
                else [outputs["lexical_weights"]]
            )

        if "colbert_vecs" in outputs and outputs["colbert_vecs"] is not None:
            response["colbert_vecs"] = (
                [vec.tolist() for vec in outputs["colbert_vecs"]]
                if isinstance(outputs["colbert_vecs"], list)
                else [outputs["colbert_vecs"].tolist()]
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {str(e)}"
        )


@app.post("/compute_score", response_model=ScoreResponse)
async def compute_relevance_scores(request: TextPairRequest):
    """문장 쌍 간의 유사도 점수 계산"""
    if global_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 문장 쌍 형식 변환
        sentence_pairs = [(pair[0], pair[1]) for pair in request.sentence_pairs]

        # 점수 계산
        scores = global_model.compute_score(
            sentence_pairs=sentence_pairs,
            batch_size=request.batch_size,
            max_query_length=request.max_query_length,
            max_passage_length=request.max_passage_length,
            weights_for_different_modes=request.weights_for_different_modes,
        )

        # 응답 형식으로 변환
        return {
            "colbert": scores.get("colbert"),
            "sparse": scores.get("sparse"),
            "dense": scores.get("dense"),
            "sparse_dense": scores.get("sparse+dense"),
            "colbert_sparse_dense": scores.get("colbert+sparse+dense"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Score computation failed: {str(e)}"
        )


@app.get("/_health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "model_loaded": global_model is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
