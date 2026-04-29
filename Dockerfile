FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt

ARG MODEL_NAME=sambus211/zhaw_at_touche_setup6_qwen

RUN python3 - <<PY
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "${MODEL_NAME}"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
PY

ENTRYPOINT ["python3", "/predict.py"]
