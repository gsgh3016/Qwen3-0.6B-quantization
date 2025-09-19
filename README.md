# Qwen3-0.6B Quantization Experiment

이 프로젝트는 Qwen3-0.6B 모델의 양자화 실험을 위한 프레임워크입니다. 다양한 양자화 방법을 적용하고 원본 모델과 성능을 비교합니다.

## 프로젝트 구조

```
Qwen3-0.6B-quantization/
├── models/
│   ├── original/                    # 원본 모델
│   ├── quantized/                   # 양자화된 모델들
│   │   ├── int8/                   # INT8 동적 양자화
│   │   ├── int4/                   # INT4 정적 양자화
│   │   └── dynamic/                # 동적 양자화
│   └── experiments/                # 실험 결과
├── experiments/                     # 실험 스크립트
│   ├── quantization_experiments.py
│   ├── accuracy_comparison.py
│   └── performance_benchmark.py
├── quantization/                    # 양자화 관련 코드
│   ├── quantizer.py
│   └── quantized_model.py
├── evaluation/                      # 평가 관련 코드
│   ├── accuracy_evaluator.py
│   ├── performance_evaluator.py
│   ├── dataset_loader.py           # 데이터셋 로딩
│   └── evaluation_results.py       # 평가 결과 관리
├── logs/                           # 로깅 및 출력 관리
│   ├── logger.py
│   ├── progress.py
│   └── result_printer.py
├── schemas/                        # 데이터 스키마
│   ├── model_schemas.py
│   ├── prediction_schemas.py
│   └── schema_builder.py
├── quantization_evaluation.py      # 양자화 평가 메인 함수
├── model_manager.py                # 모델 관리
├── quantize.py                     # 양자화 실행 스크립트
├── evaluate.py                     # 모델 평가 스크립트
└── configs/
    └── configs.yaml
```

## 주요 기능

### 1. 양자화 방법

- **INT8 동적 양자화**: 런타임에 동적으로 양자화
- **INT4 정적 양자화**: 사전에 정적으로 양자화
- **동적 양자화**: 일반적인 동적 양자화

### 2. 평가 메트릭

- **정확도 비교**: 원본 모델 대비 정확도 변화
- **성능 벤치마크**: 추론 속도, 메모리 사용량, 처리량
- **모델 크기 비교**: 양자화 전후 모델 크기 변화
- **데이터셋 기반 평가**: configs.yaml에서 지정한 데이터셋으로 평가
- **Perplexity 평가**: 모델의 언어 모델링 성능 측정

### 3. 실험 기능

- **자동화된 실험**: 여러 양자화 방법을 자동으로 테스트
- **결과 저장**: JSON 형태로 실험 결과 저장
- **로깅 시스템**: 체계적인 로그 관리 및 진행 상황 추적
- **시각화**: 성능 비교 차트 생성
- **데이터셋 자동 로딩**: Hugging Face 데이터셋 자동 다운로드 및 로딩
- **평가 결과 관리**: 체계적인 결과 저장, 로딩, 비교 기능

### 4. 로깅 시스템

- **중앙화된 로깅**: 모든 로그를 `logs/` 디렉토리에서 관리
- **진행 상황 추적**: 실시간 진행률 표시
- **결과 출력**: 깔끔한 결과 출력 및 포맷팅
- **에러 처리**: 체계적인 에러 로깅 및 처리

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -e .
```

### 2. 설정 파일 수정

`configs/configs.yaml` 파일에서 실험 설정을 조정할 수 있습니다:

```yaml
quantization_experiment:
  model:
    original:
      name: "Qwen/Qwen3-0.6B"
      cache_dir: "./models/original"
  quantization_methods:
    - name: "int8"
      bits: 8
      method: "dynamic"
    - name: "int4"
      bits: 4
      method: "static"
  evaluation:
    dataset: "microsoft/xglue"
    metrics: ["accuracy", "perplexity", "latency"]
```

### 3. 실험 실행

#### 전체 실험 파이프라인 실행

```bash
# 전체 실험 실행 (양자화 + 평가)
python main.py
```

#### 개별 단계 실행

```bash
# 양자화만 실행
python -c "from main import quantize; from logs.logger import QuantizationLogger; quantize(QuantizationLogger())"

# 평가만 실행
python -c "from evaluation.accuracy_evaluator import evaluate_before_quantization, evaluate_after_quantization; from logs.logger import QuantizationLogger; logger = QuantizationLogger(); evaluate_before_quantization(logger); evaluate_after_quantization(logger)"
```

#### 테스트 실행

```bash
# 평가 기능 테스트
python test_evaluation.py
```

#### 개별 실험 실행

```bash
# 정확도 비교만 실행
python -m experiments.accuracy_comparison

# 성능 벤치마크만 실행
python -m experiments.performance_benchmark
```

## 사용 예시

### 데이터셋 기반 평가

```python
from evaluation.dataset_loader import DatasetLoader
from evaluation.accuracy_evaluator import AccuracyEvaluator
from core.model_manager import load_original_model, load_original_tokenizer

# 데이터셋 로딩
dataset_loader = DatasetLoader()
prompts, expected = dataset_loader.load_evaluation_dataset()

# 모델 로딩 및 평가
model = load_original_model()
tokenizer = load_original_tokenizer()
evaluator = AccuracyEvaluator(tokenizer)

# 정확도 평가
results = evaluator.evaluate_model_accuracy(
    model=model,
    test_prompts=prompts,
    quantization_method="original"
)
```

### 평가 결과 관리

```python
from evaluation.evaluation_results import EvaluationResults

# 결과 관리자 생성
results_manager = EvaluationResults()

# 결과 저장
results_file = results_manager.save_evaluation_results(
    original_results=original_results,
    quantized_results=quantized_results,
    dataset_name="microsoft/xglue",
    evaluation_type="comprehensive"
)

# 결과 출력
results_manager.print_evaluation_summary(
    original_results=original_results,
    quantized_results=quantized_results,
    dataset_name="microsoft/xglue"
)
```

### 양자화 실행

```python
from quantization import Quantizer
from logs.logger import QuantizationLogger

logger = QuantizationLogger()
quantizer = Quantizer(
    model_path="./models/original",
    output_dir="./models/quantized",
    logger=logger
)

# INT8 동적 양자화
quantizer.quantize_int8_dynamic()

# INT4 정적 양자화
quantizer.quantize_int4_static()
```

### 성능 비교

```python
from experiments.accuracy_comparison import compare_accuracy
from logs.logger import QuantizationLogger

# 정확도 비교
logger = QuantizationLogger()
results = compare_accuracy(
    test_prompts=["1+1=", "The capital of France is"],
    quantization_methods=["int8", "int4"]
)
```

## 실험 결과

실험 결과는 `models/experiments/` 디렉토리에 JSON 형태로 저장됩니다:

- `quantization_experiment_results.json`: 양자화 실험 결과
- `accuracy_comparison_results.json`: 정확도 비교 결과
- `performance_benchmark_results.json`: 성능 벤치마크 결과

## 주요 클래스

### Quantizer

양자화를 담당하는 메인 클래스입니다.

### QuantizedModelManager

양자화된 모델들을 관리하고 로드하는 클래스입니다.

### AccuracyEvaluator

정확도 평가를 담당하는 클래스입니다.

### PerformanceEvaluator

성능 평가를 담당하는 클래스입니다.

## 설정 옵션

### 양자화 방법 설정

- `name`: 양자화 방법 이름
- `bits`: 비트 수 (4, 8 등)
- `method`: 양자화 방법 ("static", "dynamic")

### 평가 설정

- `dataset`: 평가용 데이터셋
- `metrics`: 평가 메트릭 목록
- `temperature`: 샘플링 온도
- `top_k`: 상위 k개 토큰

### 실험 설정

- `save_results`: 결과 저장 여부
- `results_dir`: 결과 저장 디렉토리
- `compare_original`: 원본 모델과 비교 여부

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
