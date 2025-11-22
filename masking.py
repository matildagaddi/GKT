###################
# masking stage 0
###################
import argparse
import torch
import os
import jsonlines
import json
import re
import random
import time
from tqdm import tqdm
from utils_data import Gsm8k_dataset,CSQA_dataset,AQuA_dataset,SecQA_dataset,Stage0_CyberNERQA_dataset,CyberNERQA_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from thop import profile
from peft import PeftModel
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import DataCollatorWithPadding
logging.basicConfig(level=logging.INFO)

# generate new "Masked_Question" column using "Question" column from CyberNERQA data
# use system prompt

# load data


# generate new "Masked_Question" column and save to new json copy

# save file of new "Masked_Question"



# measure accuracy of masking
### exact string matching with "PII-free Question" column. strip whitespace, lower
### check no row["Entity"]["IDENTITY"] (list) are in masked question 


# stage 1 will only parse through new JSON with "Masked_Question" column, stage 2 will use raw "Question" column

def answer_cleansing_mask(pred):
    """Extract masked question from model response."""
    
    # Split by "Masked Question:" and take the second occurrence (the actual answer, not the example)
    parts = pred.split("Masked Question:")
    
    if len(parts) > 1:
        # Get the "Masked Question:" section (skip the example)
        masked_section = parts[2].strip()
        
        # Find the first question mark and cut there
        question_mark_idx = masked_section.find('?')
        if question_mark_idx != -1:
            masked_q = masked_section[:question_mark_idx + 1].strip()
        else:
            # No question mark found, take first line or sentence
            masked_q = masked_section.split('\n')[0].strip()
            # Remove common trailing artifacts
            masked_q = re.sub(r'\s*(Answer|Question|Example).*$', '', masked_q, flags=re.IGNORECASE).strip()
        
        return masked_q
    
    # Fallback: no "Masked Question:" found
    # Look for the last sentence ending with '?'
    sentences = re.split(r'(?<=[.!?])\s+', pred)
    for sent in reversed(sentences):
        if '?' in sent:
            return sent.strip()
    
    return pred.strip()  # Last resort


def generate(
    model,
    input_data,args
):
    top_p= 0.3 #was 0.9 # want masking to be accurate, not creative
    temp=0.2 # temp was 0.8, changed to handle "Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed."
    max_gen_len = args.max_gen_len
    for i in input_data:
        input_data[i]=input_data[i].squeeze(1) 
       
    output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len, temperature = temp, top_p = top_p) 
    results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results

def generation_loop(dataloader, model,args): #TODO reformat for masking
    results=[]
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(model.device)

        with torch.no_grad():
            generated_tokens = generate(model, 
                                        batch_data,args
                                        )
            for i in generated_tokens:
                #print(i)
                results.append(i.strip())
        
    return results 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-70b-hf","meta-llama/Llama-2-13b-hf","google/flan-t5-xl","bigscience/bloom-7b1"], default="meta-llama/Llama-2-13b-hf")#default="meta-llama/Llama-2-13b-hf")
    #parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_batch_size', type=int, default=12) #4
    parser.add_argument('--dataset', choices=["Stage0_CyberNERQA"]) 
    # removed --if_concise_prompt
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--out_path', type=str, default="data/CyberNERQA_masked/")
    parser.add_argument('--max_gen_len', type=int, default=50) # might need to be larger, refine best option
    parser.add_argument('--batch_size', type=int, default=24)
    #parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7 CNNDM:0") #maybe add fewshot later if accuracy is bad
    args = parser.parse_args()

    
    ### save to a new json dataset with the new column? another dataloader used by stage 1 and 2
    folder_name = "data/CyberNERQA_masked/"
    base_data_name = "CyberNERQA_masked"
    counter = 1
    masked_data_path = os.path.join(folder_name,f"{base_data_name}_{counter}.json")
    while os.path.isfile(masked_data_path):
        counter += 1
        masked_data_path = os.path.join(folder_name, f"{base_data_name}_{counter}.json")
    print(f"Output masked json data will be saved to: {masked_data_path}")
    print("Set the Dataloader's data_path to this path ^")
    


    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    if "t5" in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto")
    elif "70b" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,load_in_8bit=True, torch_dtype="auto", device_map="auto",cache_dir="./cache") 
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="sequential", #auto
                                                     low_cpu_mem_usage=True, trust_remote_code=True)
        # was torch_dtype=torch.float16, changed for lower GPU memory
        original_named_parameters = dict(model.named_parameters())
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    if args.dataset =="Stage0_CyberNERQA":
        dataset=Stage0_CyberNERQA_dataset(tokenizer,args)
    
    total=len(dataset)
    right_match=0
    right_clean=0
    empty_masked_q=0

    #with jsonlines.open(masked_data_path, mode='w') as writer:
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
    # Measure FLOPs
    if "t5" not in args.model_name:
        for batch_data in dataloader:
            flops, params = profile(model, inputs=(batch_data["input_ids"].squeeze(1).to(model.device),))
            print(f"FLOPs: {flops / 1e9} G FLOPs")  # FLOPs in billions (GFLOPs)
            print(f"Number of parameters: {params / 1e6} M")  # Parameters in millions (M)
            break
    else:
        flops=0
        params=0
    # recode start time
    start_time = time.time()
    masked_qs=generation_loop(dataloader, model,args)
    
    # end time
    end_time = time.time()
    execution_time = end_time - start_time
    print("-------------")
    print("!!! execution_time",execution_time)
    print("-------------")

    masked_data = []
    for qid, maskq in enumerate(masked_qs):
        raw_data_row = dataset.data[qid] #original row of data, then add later the "Masked_Question" column.
        
        if isinstance(raw_data_row, str):
            raw_data_row = json.loads(raw_data_row)
        elif not isinstance(raw_data_row, dict):
            raise TypeError(f"Unexpected row type for index {qid}: {type(raw_data_row)}")

        item = dict(raw_data_row)
        final_masked_q = answer_cleansing_mask(maskq)
        item["Masked_Question"] = final_masked_q

        masked_data.append(item)
        
        # accuracy checks
        ### Check for exact match
        if str(final_masked_q).strip().lower() == str(raw_data_row.get("PII-free Question", "")).strip().lower():
            right_match += 1

        ### Check for at least entities being masked/removed
        # Get all entity values from all entity types
        all_entity_values = []
        for values in raw_data_row.get("Entities", {}).values():
            if isinstance(values, list):
                all_entity_values.extend(values)

        if not any(str(entity_val) in str(final_masked_q) for entity_val in all_entity_values):
            right_clean += 1        
        # if not any(str(entity) in str(final_masked_q) for entity in raw_data_row.get("Entities", [])):
        #     right_clean += 1

        ####################################
        ### TODO: collect metrics like a classifier (precision, recall, f1) tokenized PII vs non-PII
        ####################################
        
        
        if len(str(final_masked_q).strip()) <= 2:
            empty_masked_q += 1
    
    with open(masked_data_path, "w") as f:
        json.dump(masked_data, f, indent=2)
                
    

    ####save args
    args_dict = vars(args)
    args_dict["Execution time"]=execution_time
    args_dict["right exact match"]=right_match
    args_dict["right clean, no PII"]=right_clean
    args_dict["empty masked questions"]=empty_masked_q
    args_dict["total"]=total
    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    args_dict["timestamp"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["masked data path"] = masked_data_path
    args_dict["acc_match"]=right_match/total
    args_dict["acc_clean"]=right_clean/total
    print("-------------")
    print("!!! right match: ",right_match, "    right clean: ",right_clean, "   total: ", total, "    empty masked questions:", empty_masked_q)
    print("!!! accuracy match: ", right_match/total, "!!! accuracy clean: ", right_clean/total)
    print("-------------")
    
    
    #json_path = os.path.join(folder_name,str(args.max_gen_len)+"_args.json")
    basej_name = str(args.max_gen_len)+"_args"
    json_path=os.path.join(folder_name,f"{basej_name}.json")
    # if os.path.isfile(out_path):
    #     assert False
    counter = 1
    # Find next available filename
    while os.path.isfile(json_path):
        json_path = os.path.join(folder_name, f"{basej_name}_{counter}.json")
        counter += 1
    
    print(f"Arg MASKING stage will be saved to: {json_path}")
    print("------------------------------ END MASKING ------------------------------")
    
    with open(json_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)