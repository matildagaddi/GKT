#stage 1
import argparse
import torch
import os
import jsonlines
import json
import re
import random
import time
from tqdm import tqdm
from utils_data import CyberNERQA_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from thop import profile
from peft import PeftModel
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
logging.basicConfig(level=logging.INFO)

# MAKE A CLASS TO CALL FROM MAIN
# RETURN/SAVE output paths and results especially to feed to stage 2

# COLLECT ACCURACY OF SEMANTIC CLASSIFICATION

def stage1(model_name, dataset="CyberNERQA",
                 batch_size=12, max_gen_len=30,
                 data_path="./data/", out_path="output/big2small/", 
                 masking=1, few_shot=1)

        args_dict = {}
        args_dict["model_name"] = model_name
        args_dict["dataset"] = dataset
        args_dict["batch_size"] = batch_size
        args_dict["max_gen_len"] = max_gen_len
        args_dict["data_path"] = data_path
        args_dict["out_path"] = out_path
        args_dict["masking"] = masking
        args_dict["few_shot"] = few_shot
        
        model = None
        tokenizer = None
        #args_dict = vars(args) #how to do this efficiently with function args


    #helper functions
    def generate(input_data, max_gen_len, model):
        top_p= 0.9
        for i in input_data:
            input_data[i]=input_data[i].squeeze(1) 
           
        output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len, temperature = 0.2,top_p = top_p) # temp was 0.8, changed to handle "Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed."
        results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return results
    
    def generation_loop(dataloader, model):
        results=[]
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(model.device)
            
            with torch.no_grad():
                generated_tokens = model.generate(batch_data)
                
                for i in generated_tokens:
                    results.append(i.split("\nA:")[-1].strip())
            
        return results
        
    # Save output
    args_out_path=os.path.join(self.out_path,"stage1")
    model_temp=self.model_name.split("/")[-1]
    folder_name = os.path.join(args_out_path,model_temp,self.dataset)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"folder '{folder_name}' created")
    base_name = "output_stage1_fs"+{few_shot}+"_msk"+{stage1_args["masking"]}
    out_path=os.path.join(folder_name,f"1_{base_name}.jsonlines")
    # Find next available filename
    counter = 1
    while os.path.isfile(out_path):
        out_path = os.path.join(folder_name, f"{counter}_{base_name}.jsonlines")
        counter += 1
    print("-------------")
    print(f"Output will be saved to: {out_path}")
    print("Use this path for stage 2 big_output_path ^")
    print("-------------")

    # Save args
    #json_path = os.path.join(folder_name,str(args.max_gen_len)+"_args.json")
    basej_name = "args_stage1_fs"+{few_shot}+"_msk"+{stage1_args["masking"]}
    json_path=os.path.join(folder_name,f"{basej_name}.json")
    # if os.path.isfile(out_path):
    #     assert False
    counter = 1
    # Find next available filename
    while os.path.isfile(json_path):
        json_path = os.path.join(folder_name, f"{basej_name}_{counter}.json")
        counter += 1
    
    print(f"Args stage 1 will be saved to: {json_path}")
    



    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    elif "70b" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=True, torch_dtype="auto", device_map="auto",cache_dir="./cache") 
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="sequential", #auto
                                                     low_cpu_mem_usage=True, trust_remote_code=True)
        # was torch_dtype=torch.float16, changed for lower GPU memory
        original_named_parameters = dict(model.named_parameters())
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if dataset =="SecQA":
        dataset=SecQA_dataset(tokenizer, out_path, dataset, stage2=False)
    elif dataset =="CyberNERQA":
        dataset=CyberNERQA_dataset(tokenizer,args,stage2=False) #need to pass args for initializing class, dictionary?

    with jsonlines.open(out_path, mode='w') as writer:
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
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


        ## add classification in dataloader, split train and test closer to 50/50
        # add cloud-edge classification (check semantic similarity to questions that failed to be masked, if similar enough, don’t send question to cloud, stay on the edge)
        ## need to keep track of those questions and their masking success, maybe indexes
        # figure out batching, maybe separate into two new dataloading sets (one for edge and one for cloud) to batch from there?
        # append empty string "" as cloud guidance if classified as staying on edge to keep compatible with rest of pipeline
        # collect runtime of classification
        
        guidances=generation_loop(dataloader, model,args)
        # end time
        end_time = time.time()
        execution_time = end_time - start_time
        print("-------------")
        print("!!!execution_time",execution_time)
        print("-------------")
        qid=0
        for qid, ans in enumerate(guidances):
            item={}
            item[qid]=ans
            writer.write(item)
            qid+=1

    
    ####save args and output
    print(locals())
    args_dict = locals()#function inputs
    args_dict["Execution time"]=execution_time
    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    args_dict["timestamp"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["out_path"] = out_path
    
    
    with open(json_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)


    return args_dict, guidances

        

## backwards comaptibility
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, choices=["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-70b-hf","meta-llama/Llama-2-13b-hf","google/flan-t5-xl","bigscience/bloom-7b1"], default="meta-llama/Llama-2-13b-hf")#default="meta-llama/Llama-2-13b-hf")
#     parser.add_argument('--max_seq_len', type=int, default=1024)
#     parser.add_argument('--max_batch_size', type=int, default=12) #4
#     parser.add_argument('--dataset', choices=['GSM8K', 'CSQA',"CNNDM","AQuA","SecQA","CyberNERQA"])
#     parser.add_argument('--if_concise_prompt', choices=[False,"Provide the answer in a brief manner: ","Provide a brief hint for the question: "],default=False)
#     parser.add_argument('--data_path', type=str, default="./data/")
#     parser.add_argument('--out_path', type=str, default="output/big2small/")
#     parser.add_argument('--max_gen_len', type=int, default=30)
#     parser.add_argument('--batch_size', type=int, default=24)
#     parser.add_argument('--masking', type=int, choices=[0,1])
#     parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7 CNNDM:0")
#     args = parser.parse_args()