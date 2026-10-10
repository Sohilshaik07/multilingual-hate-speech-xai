from app.utils.model_loader import (
    load_cached_pipeline,
    predict_sample,
    get_model_identifier,
)
from app.utils.labels import get_labels_for_dataset

key = "hecm"
model, tokenizer, device = load_cached_pipeline(key, device="cpu")

print("\n--- CHECKPOINT DIAGNOSTICS ---")
print("Model identifier:", get_model_identifier(key))
print("Tokenizer:", tokenizer.name_or_path)
print("Model classes:", model.config.num_labels)
print("Model id2label:", model.config.id2label)
print("Model label2id:", model.config.label2id)
print("Project label mapping:", get_labels_for_dataset(key))
print("Device:", device)

samples = [
    "hello",
    "Aaj mausam bahut achha hai.",
    "Modi ji ne farmers ke liye kafi policy banayi hain par implementation thoda slow hai.",
]

for text in samples:
    result = predict_sample(model, tokenizer, text, key, device=device)
    print("\nInput:", text)
    print("Predicted class index:", result["predicted_class"])
    print("Predicted label:", result["label_mapping"].get(
        result["predicted_class"], "UNKNOWN"
    ))
    print("Probabilities:", result["probabilities"])