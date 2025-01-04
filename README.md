# Simple-bge-server

## Command
```
uv run main.py
```

## API Specification
### 기본 정보
- Base URL: `http://[hostname]:8000`
- 서비스명: M3 Embedding Service
- 사용 모델: BAAI/bge-m3

### Endpoints

#### 1. 텍스트 임베딩 생성
* POST /embed

텍스트를 입력받아 임베딩 벡터를 생성합니다.

**Request Body**
```json
{
    "texts": ["string"],
    "instruction": "string (optional)",
    "return_dense": true (optional),
    "return_sparse": false (optional),
    "return_colbert_vecs": false (optional),
    "batch_size": 256 (optional),
    "max_length": 512 (optional)
}
```

**Response**
```json
{
    "dense_vecs": [[float]] (optional),
    "lexical_weights": [{"word": float}] (optional),
    "colbert_vecs": [[[float]]] (optional)
}
```

**Parameters 설명**
- texts: 임베딩을 생성할 텍스트 문장들의 리스트
- instruction: 임베딩 생성 시 사용할 추가 지시사항 (선택)
- return_dense: 밀집 벡터 반환 여부
- return_sparse: 희소 벡터 반환 여부
- return_colbert_vecs: ColBERT 벡터 반환 여부
- batch_size: 배치 크기
- max_length: 최대 텍스트 길이

#### 2. 문장 유사도 점수 계산
* POST /compute_score

두 문장 간의 유사도 점수를 계산합니다.

**Request Body**
```json
{
    "sentence_pairs": [["query", "passage"]],
    "weights_for_different_modes": [float] (optional),
    "batch_size": 256 (optional),
    "max_query_length": 512 (optional),
    "max_passage_length": 512 (optional)
}
```

**Response**
```json
{
    "colbert": [float] (optional),
    "sparse": [float] (optional),
    "dense": [float] (optional),
    "sparse_dense": [float] (optional),
    "colbert_sparse_dense": [float] (optional)
}
```

**Parameters 설명**
- sentence_pairs: 유사도를 계산할 문장 쌍들의 리스트
- weights_for_different_modes: 각 모드별 가중치
- batch_size: 배치 크기
- max_query_length: 쿼리 문장의 최대 길이
- max_passage_length: 패시지 문장의 최대 길이

#### 3. 헬스 체크
* GET /_health

서버와 모델의 상태를 확인합니다.

**Response**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 오류 응답
서비스는 다음과 같은 HTTP 상태 코드를 반환할 수 있습니다:

- 500: 서버 내부 오류
  - 모델이 로드되지 않은 경우
  - 임베딩 생성 실패
  - 점수 계산 실패

## 주의사항
1. 서버 시작 시 자동으로 모델이 로드됩니다.
2. GPU(CUDA:0)를 사용하며, FP16 최적화가 적용됩니다.
3. 모든 임베딩은 정규화되어 반환됩니다.
