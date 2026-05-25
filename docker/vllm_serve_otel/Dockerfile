FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
      software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa \
      && apt-get update \
      && apt-get install -y python3.10 python3.10-distutils python3.10-venv curl git cargo build-essential \
      && rm -rf /var/lib/apt/lists/*

# Make python3 → python3.10 (no otel semantic conventions otherwise)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip install vllm transformers requests bitsandbytes \
    && pip install \
         'opentelemetry-sdk>=1.26.0,<1.27.0' \
         'opentelemetry-api>=1.26.0,<1.27.0' \
         'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' \
         opentelemetry-instrumentation-fastapi \
    && pip install 'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0' 


ENV VLLM_LOGGING_LEVEL=DEBUG
EXPOSE 8000

CMD ["opentelemetry-instrument", \
     "vllm", "serve", \
     "/model", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--otlp-traces-endpoint", "grpc://jaeger:4317"]
