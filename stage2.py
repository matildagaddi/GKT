#stage 2
import argparse
import torch
import os
import jsonlines
import json
import re
import random
from utils_data import CyberNERQA_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
import time
from tqdm import tqdm
from thop import profile
import evaluate
from datetime import datetime
from zoneinfo import ZoneInfo
from utils import *


# MAKE A CLASS TO CALL FROM MAIN
# RETURN/SAVE output paths and results


def answer_cleansing_cybernerqa(pred,few_shot):
    pattern = re.compile(r'The answer is ([A-Z])')
    res = pattern.findall(pred)
    if len(res) >= few_shot+1:
        return res[few_shot].upper()  # 'A', 'B', ...
    else:
        option=["A","B","C","D"]
        return option[random.randint(0,3)]

# def generate(model, tokenizer, input_data, max_gen_len):
#     top_p= 0.95
#     max_gen_len = max_gen_len
#     for i in input_data:
#         input_data[i]=input_data[i].squeeze(1)
#     output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len,temperature = 0.4,top_p = top_p) #do_sample=True)#, top_p=top_p) #temp was 0.8
#     results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
#     return results

# def generation_loop(dataloader, model, tokenizer, dataset):
#     results=[]
#     for batch_data in tqdm(dataloader):
#         batch_data = batch_data.to(model.device)
#         #batch_data = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch_data.items()}
#         with torch.no_grad():
#             generated_tokens = generate(model, tokenizer, batch_data, max_gen_len)
#             #print(generated_tokens)
#             for i in generated_tokens:
#                 if dataset =="CyberNERQA":
#                     answer=answer_cleansing_cybernerqa(i,few_shot)
#                 elif dataset =="SecQA":
#                     answer=answer_cleansing_secqa(i,few_shot)
                
#                 item={"output":i,"cot":decoded_output,"ans":answer}
#                 results.append(item)
#     return results


def stage2(model_name, stage1_args, big_output_path, mc_data_path="./data/", dataset="CyberNERQA", out_path="output/big2small/", batch_size=5, max_gen_len=100, few_shot=1):#guidances, #instead of big_output_path, pass directly instead of reading from a file 
    
    args_dict = {}
    args_dict["model_name"] = model_name
    args_dict["dataset"] = dataset
    args_dict["batch_size"] = batch_size
    args_dict["max_gen_len"] = max_gen_len
    args_dict["data_path"] = mc_data_path
    args_dict["out_path"] = out_path
    args_dict["few_shot"] = few_shot
    
    # Save output
    model_temp=model_name.split("/")[-1]
    folder_name = os.path.join(out_path,'stage2',model_temp,dataset)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"folder '{folder_name}' created")
    # stage1_len=stage1_args['max_gen_len'] #big_output_path.split("/")[-2] 
    base_name = f"output_stage2_fs{few_shot}_msk{stage1_args['masking']}"
    out_path=os.path.join(folder_name,f"{base_name}.jsonlines")
    # Find next available filename
    counter = 1
    while os.path.isfile(out_path):
        out_path = os.path.join(folder_name, f"{counter}_{base_name}.jsonlines")
        counter += 1
    print(f"Output will be saved to: {out_path}")

    
    # Save args
    #json_path = os.path.join(folder_name,str(max_gen_len)+"_args.json")
    basej_name = f"args_stage2_fs{few_shot}_msk{stage1_args['masking']}"
    json_path = os.path.join(folder_name,f"{basej_name}.json")
    # Find next available filename
    counter = 1
    while os.path.isfile(json_path):
        json_path = os.path.join(folder_name, f"{counter}_{basej_name}.json")
        counter += 1
    print(f"Args stage 2 will be saved to: {json_path}")


    

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    elif "70b" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=True, torch_dtype="auto", device_map="auto",cache_dir="./cache")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                                    low_cpu_mem_usage=True, trust_remote_code=True)
    #dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collote_fn)
    
    # Define PAD Token = EOS Token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    # if dataset =="SecQA":
    #     dataset=SecQA_dataset(tokenizer,args,step='answer')
    # elif dataset =="CyberNERQA":
    dataset=CyberNERQA_dataset(tokenizer, data_path=mc_data_path, args={'few_shot': few_shot, 'big_output_path': big_output_path},step='answer')

    total=len(dataset)
    right=0
    
    with jsonlines.open(out_path, mode='w') as writer:
        dataloader = DataLoader(dataset, batch_size=batch_size)
        # Measure FLOPs
        if "t5" not in model_name:
            for batch_data in dataloader:
                flops, params = profile(model, inputs=(batch_data["input_ids"].squeeze(1).to(model.device),))
                print(f"FLOPs: {flops / 1e9} G FLOPs")  # FLOPs in billions (GFLOPs)
                print(f"Number of parameters: {params / 1e6} M")  # Parameters in millions (M)
                break
        else:
            flops=0
            params=0

        def stage2_postprocess(raw):
            ans = answer_cleansing_cybernerqa(raw, few_shot)
            return {"output": raw, "cot": None, "ans": ans}
        # recode start time
        start_time = time.time()
        answers=generation_loop(dataloader, model, tokenizer, max_gen_len=max_gen_len, postprocess_fn=stage2_postprocess)
        # end time
        end_time = time.time()
        execution_time = end_time - start_time
        print("-------------")
        print("!!! execution_time",execution_time)
        print("-------------")

        wrong_qs = {}
        for qid, ans in enumerate(answers):
            item={}
            item[qid]=ans
            writer.write(item)
            if str(ans["ans"])==str(dataset.data[qid]["answer"]):
                right+=1
            else:
                wrong_qs[qid]=[dataset.data[qid], ans["ans"]]
                

    ####save args and outputs
    #### also save stage 1 model and args? in own sub dictionary?
    args_dict["Execution time"]=execution_time
    args_dict["right"]=right
    args_dict["total"]=total
    args_dict["timestamp"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["text path"] = out_path
    
    args_dict["acc"]=right/total
    args_dict["wrong"]=wrong_qs
    print("!!! right: ",right, "   total: ", total, "   accuracy: ", right/total)

    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    args_dict["stage 1 args"]=stage1_args

    with open(json_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
    print("------------------------------ END ------------------------------")

    return args_dict
    