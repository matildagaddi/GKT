# main file to run all steps at once

### TODO: 
# finish classify function
# figure out dataloader (new one for each step or parameters in a catch all one)
# figure out passing data from one step to next
# add edge model in stage 1
# rename repo

### Stage 0 (mask data and train classifier)
# Mask training set data (masking script should work for train and test, not just stage 0)
# Assess which ones failed and save: set labeled Qs to their failed_embs or success_embs for cosine similarity in testing
# Report masking performance on train set (F1, recall)
#CUDA_VISIBLE_DEVICES=0 python masking.py --model_name "meta-llama/Llama-2-7b-hf" --batch_size 50 --max_gen_len 50 --dataset Stage0_CyberNERQA --train_test "train" --split_train 0.5 

## Stage 0.5 (classify and mask new data)
# Classify new testing set data questions based on semantic similarity to failed_embs and success_embs
        # could use failed accuaracy match or failed accuracy clean depending on class balance
# Report classification performance (accuracy)
# Apply masking to those classified as cloud

### Stage 1 (cloud guidance)
# send masked questions to cloud and non masked to edge (masking file needs to work in both train and test scenarios, param)
# Report masking performance on test set (F1, recall)
#CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py --model_name "meta-llama/Llama-2-13b-hf" --batch_size 10 --max_gen_len 30 --dataset CyberNERQA --data_args '/home/jovyan/GKT/data/CyberNERQA_masked/50_args.json' --masking 1 --few_shot 1
# should call masking  CUDA_VISIBLE_DEVICES=0 python masking.py --model_name "meta-llama/Llama-2-7b-hf" --batch_size 50 --max_gen_len 50 --dataset Stage0_CyberNERQA --train_test "test" --split_train 0.5 --embeddings 
# how can this be a class/function that I can just pass the embeddings into?

### Stage 2 (edge answer)
# Edge makes final QA answer with raw context and guidance from stage 1
# Report performance on QA set (accuracy)
#CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py --model_name "meta-llama/Llama-2-7b-hf" --dataset CyberNERQA --data_path '/home/jovyan/GKT/data/CyberNERQA_masked/CyberNERQA_masked_1.json' --max_gen_len 100 --out_path "output/big2small" --big_output_path "output/big2small/stage1/Llama-2-13b-hf/CyberNERQA/30_output_stage1_9.jsonlines" --few_shot 1




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from masker import mask
from stage1 import stage1
from stage2 import stage2
from classifier import classifier
import argparse
import json

# stage 0 mask
masker_args, masked_data, fail_success_split_qs = mask(dataset="CyberNERQA")

# stage 0.5 classify
#split and classify new data
classified_masked_test_data = classify(masked_test_data, fail_success_split_qs, train_split=0.5, fail_split='match')
# go in and add column "send_to_cloud" = 1 or 0

# stage 1 guide
stage1_args, guidances = stage1(dataset=CyberNERQA_masked_classified, cloud_model="meta-llama/Llama-2-13b-hf", edge_model="meta-llama/Llama-2-7b-hf") #either new dataloader for _masked_classified OR make CyberNERQA loader know when to return CyberNERQA_masked_classified with new parameter? Which is better?

# stage 2 answer
stage2_args = stage2(guidances=guidances, stage1_args = stage1_args, dataset=CyberNERQA_masked_classified, model_name="meta-llama/Llama-2-7b-hf")

print(stage2_args)



# if __name__ == "__main__":
#     main()