# main file to run all steps at once
from masking import MaskingPipeline

mp = MaskingPipeline(
    model_name="meta-llama/Llama-2-7b-hf",
    batch_size=50,
    max_gen_len=50,
)

# Stage 0 (train) → produces masked JSON, metrics, and failed/success embeddings
train_output = mp.run(
    dataset_path="train.json",
    split="train",
    save_prefix="Stage0_CyberNERQA"
)

# Stage 1 (test) → pass embeddings into masking
test_output = mp.run(
    dataset_path="test.json",
    split="test",
    failed_embs=train_output["failed_embs"],
    success_embs=train_output["success_embs"]
)

### Stage 0
# Mask training set data (masking script should work for train and test, not just stage 0)
# Assess which ones failed and save: set labeled Qs to their failed_embs or success_embs for cosine similarity in testing
# Report masking performance on train set (F1, recall)
CUDA_VISIBLE_DEVICES=0 python masking.py --model_name "meta-llama/Llama-2-7b-hf" --batch_size 50 --max_gen_len 50 --dataset Stage0_CyberNERQA --train_test "train" --split_train 0.5 #want to turn into a class/function so I can just call it in this main script

### Stage 1 
# Classify new testing set data questions based on similarity to failed_embs and success_embs
# Apply masking to those classified as cloud and send them (masking file needs to work in both train and test scenarios, param)
# Otherwise get guidance from same edge model
# Report masking performance on test set (F1, recall)
CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py --model_name "meta-llama/Llama-2-13b-hf" --batch_size 10 --max_gen_len 30 --dataset CyberNERQA --data_args '/home/jovyan/GKT/data/CyberNERQA_masked/50_args.json' --masking 1 --few_shot 1
# should call masking  CUDA_VISIBLE_DEVICES=0 python masking.py --model_name "meta-llama/Llama-2-7b-hf" --batch_size 50 --max_gen_len 50 --dataset Stage0_CyberNERQA --train_test "test" --split_train 0.5 --embeddings 
# how can this be a class/function that I can just pass the embeddings into?

### Stage 2
# Edge makes final QA answer with raw context and guidance from stage 1
# Report performance on QA set (accuracy)
CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py --model_name "meta-llama/Llama-2-7b-hf" --dataset CyberNERQA --data_path '/home/jovyan/GKT/data/CyberNERQA_masked/CyberNERQA_masked_1.json' --max_gen_len 100 --out_path "output/big2small" --big_output_path "output/big2small/stage1/Llama-2-13b-hf/CyberNERQA/30_output_stage1_9.jsonlines" --few_shot 1
