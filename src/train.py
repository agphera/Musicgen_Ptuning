from transformers import AutoProcessor, MusicGenForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# 데이터 로드
def prepare_dataset(data_pairs):
    data = {"text": [], "audio": []}
    for text, audio_path in data_pairs:
        data["text"].append(text)
        data["audio"].append(audio_path)
    return Dataset.from_dict(data)

# P-Tuning 초기화 및 훈련
def train_model(data_dir, output_dir):
    data_pairs = load_data(data_dir)
    dataset = prepare_dataset(data_pairs)

    model = MusicGenForConditionalGeneration.from_pretrained("facebook/musicgen")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    train_model(data_dir="./data", output_dir="./models")
