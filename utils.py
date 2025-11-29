import argparse
# utils.py
import torch
import os
import jsonlines
import json
import re
import random
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from thop import profile
from peft import PeftModel
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
logging.basicConfig(level=logging.INFO)
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import Subset


# measure accuracy of masking
### exact string matching with "PII-free Question" column. strip whitespace, lower
### check no row["Entity"]["IDENTITY"] (list) are in masked question 

def answer_cleansing_mask(pred):
    """Extract masked question from model response."""
    
    incomplete = False

    # Split by "Masked Question:"
    parts = pred.split("Masked Question:")

    if len(parts) > 1:
        masked_section = parts[2].strip() # take the second occurrence (the actual answer, not the example)
        question_mark_idx = masked_section.find('?')

        if question_mark_idx != -1:
            return masked_section[:question_mark_idx + 1].strip(), False
        
        # No '?'
        incomplete = True
        masked_q = masked_section.split('\n')[0].strip()
        masked_q = re.sub(r'\s*(Answer|Question|Example).*$', '', masked_q, flags=re.IGNORECASE).strip()
        return masked_q, incomplete

    # Fallback: no "Masked Question:" found
    sentences = re.split(r'(?<=[.!?])\s+', pred)
    for sent in reversed(sentences):
        if '?' in sent:
            return sent.strip(), False

    # Really no '?'
    return pred.strip(), True


# stage 1 will only parse through new JSON with "Masked_Question" column, stage 2 will use raw "Question" column


# def generate(
#     model, tokenizer,
#     input_data, max_gen_len
# ):
#     top_p= 0.1 #was 0.9 # want masking to be accurate, not creative
#     temp=0.1 # temp was 0.8, changed to handle "Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed."
#     max_gen_len = max_gen_len
#     for i in input_data:
#         input_data[i]=input_data[i].squeeze(1) 
       
#     output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len, temperature = temp, top_p = top_p) 
#     results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
#     return results

# def generation_loop(dataloader, model, max_gen_len): #TODO reformat for masking
#     results=[]
#     for batch_data in tqdm(dataloader):
#         batch_data = batch_data.to(model.device)

#         with torch.no_grad():
#             generated_tokens = generate(model, tokenizer,
#                                         batch_data,
#                                         max_gen_len
#                                         )
#             for i in generated_tokens:
#                 #print(i)
#                 results.append(i.strip())
        
#     return results 

def generate_batch(model, tokenizer, input_data,
                   max_gen_len):
    """
    Pure generation step. No dataset-specific logic.
    """
    # Squeeze batch dims
    for k in input_data:
        if torch.is_tensor(input_data[k]):
            input_data[k] = input_data[k].squeeze(1)

    output = model.generate(
        **input_data,
        max_new_tokens=max_gen_len,
        top_p=0.2,
        temperature=0.2
    )
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def generation_loop(dataloader, model, tokenizer,
                    max_gen_len,
                    postprocess_fn=None,
                    device=None):
    """
    Generic dataloader → generation loop.
    `postprocess_fn` handles stage-specific answer cleaning.
    e.g., for stage2: answer_cleansing_cybernerqa
    """
    results = []
    
    for batch_data in tqdm(dataloader):
        
        # Move to device
        if device is None:
            device = model.device
        
        #batch_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}
        batch_data = batch_data.to(model.device)

        with torch.no_grad():
            raw_out = generate_batch(
                model, tokenizer,
                batch_data,
                max_gen_len
            )

        # Stage-specific behavior
        if postprocess_fn:
            processed = [postprocess_fn(x) for x in raw_out]
        else:
            processed = raw_out

        results.extend(processed)

    return results


def mask(model_name="meta-llama/Llama-2-7b-hf", dataset="CyberNERQA", raw_data_path="data/CyberNERQA_raw/CyberNERQA.json", out_path="data/CyberNERQA_masked/", max_gen_len=50, batch_size=50, few_shot=1):

    # save before reassigned
    args_dict = {}
    args_dict["model_name"]=model_name
    args_dict["dataset"]=dataset
    args_dict["max_gen_len"]=max_gen_len
    args_dict["batch_size"]=batch_size
    args_dict["few_shot"]=few_shot
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    elif "70b" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=True, torch_dtype="auto",
                                                     device_map="auto",cache_dir="./cache") 
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="sequential", #auto
                                                     low_cpu_mem_usage=True, trust_remote_code=True)
        # was torch_dtype=torch.float16, changed for lower GPU memory
        original_named_parameters = dict(model.named_parameters())
        tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
                                 
    tokenizer.padding_side = "left"  # Allow batched inference
    
    if dataset =="CyberNERQA":
        dataset=CyberNERQA_dataset(tokenizer, data_path=raw_data_path, args={'few_shot': few_shot}, step='mask') #set up this way to prep for v2
    
    total=len(dataset)
    right_match=0
    right_clean=0
    empty_masked_q=0
    tp, fp, fn, tn = 0, 0, 0, 0 

    #with jsonlines.open(masked_data_path, mode='w') as writer:
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
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
    # recode start time
    start_time = time.time()
    masked_qs=generation_loop(dataloader, model, tokenizer, max_gen_len)
    
    # end time
    end_time = time.time()
    execution_time = end_time - start_time
    print("-------------")
    print("!!! execution_time ",execution_time)
    print("-------------")

    masked_data = []
    fail_clean_qs = []
    success_clean_qs = []
    fail_match_qs = []
    success_match_qs = []
    qs_not_complete = {}
    qs_not_complete_count = 0
    for qid, maskq in enumerate(masked_qs):
        raw_data_row = dataset.data[qid] #original row of data, then add later the "Masked_Question" column.
        
        if isinstance(raw_data_row, str):
            raw_data_row = json.loads(raw_data_row)
        elif not isinstance(raw_data_row, dict):
            raise TypeError(f"Unexpected row type for index {qid}: {type(raw_data_row)}")

        item = dict(raw_data_row)
        #final_masked_q = answer_cleansing_mask(maskq)
        final_masked_q, incomplete = answer_cleansing_mask(maskq)
        if incomplete:
            qs_not_complete_count += 1
            qs_not_complete[qid] = final_masked_q
        
        item["Masked_Question"] = final_masked_q
        item['masked_clean'] = True
        item['masked_match'] = True

        
        # MATCH accuracy
        ### Check for exact match
        if str(final_masked_q).strip().lower() == str(raw_data_row.get("PII-free Question", "")).strip().lower():
            right_match += 1
            success_match_qs.append(final_masked_q)
        else:
            item['masked_match'] = False
            fail_match_qs.append(final_masked_q)

        ### CLEAN accuracy, check for at least entities being masked/removed
        # Get all entity values from all entity types
        all_entity_values = []
        for values in raw_data_row.get("Entities", {}).values():
            if isinstance(values, list):
                all_entity_values.extend(values)

        if not any(str(entity_val) in str(final_masked_q) for entity_val in all_entity_values):
            right_clean += 1 
            success_clean_qs.append(final_masked_q)
            
        else:
            #add question to list of failed masks
            fail_clean_qs.append(final_masked_q)
            item['masked_clean'] = False
            
        if len(str(final_masked_q).strip()) <= 2:
            empty_masked_q += 1


        
        ##### collect metrics like a classifier (precision, recall, f1) for tokenized PII vs non-PII
        original_q = str(raw_data_row.get("Question", "")) #*
        original_tokens = original_q.split()
        masked_tokens = str(final_masked_q).split()

        #labels for each token (1=PII, 0=non-PII)
        ground_truth = []
        for token in original_tokens:
            is_pii = any(str(entity_val) in token for entity_val in all_entity_values)
            ground_truth.append(1 if is_pii else 0)


        predictions = []
        original_lower = [t.lower() for t in original_tokens]
        masked_lower = [t.lower() for t in masked_tokens]

        
        for i, token in enumerate(original_tokens):
            token_lower = token.lower()
            # Token is "masked" (predicted as PII) if it's not in the output
            if token_lower not in masked_lower:
                predictions.append(1)  # Predicted as PII
            else:
                predictions.append(0)  # Predicted as non-PII
        
        # Update confusion matrix
        for gt, pred in zip(ground_truth, predictions):
            if gt == 1 and pred == 1:
                tp += 1  # True Positive: correctly identified PII
            elif gt == 0 and pred == 1:
                fp += 1  # False Positive: incorrectly masked non-PII
            elif gt == 1 and pred == 0:
                fn += 1  # False Negative: missed PII
            elif gt == 0 and pred == 0:
                tn += 1  # True Negative: correctly kept non-PII



        masked_data.append(item)

    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0, 4)
    accuracy_token = round((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0, 4)

    #### Save output dataset
    # save to a new json dataset with the new column? another dataloader used by stage 1 and 2
    folder_name = out_path
    base_data_name = "CyberNERQA_masked"
    counter = 1
    masked_data_path = os.path.join(folder_name,f"{base_data_name}_{counter}.json")
    while os.path.isfile(masked_data_path):
        counter += 1
        masked_data_path = os.path.join(folder_name, f"{base_data_name}_{counter}.json")
    print(f"Output masked json data will be saved to: {masked_data_path}")
    print("Set the Dataloader's data_path to this path ^")
    with open(masked_data_path, "w") as f:
        json.dump(masked_data, f, indent=2)
                
    

    ####save args 
    
    args_dict["Execution time"]=execution_time
    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    args_dict["timestamp"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["masked data path"] = masked_data_path
    args_dict["acc_match"]=right_match/total
    args_dict["acc_clean"]=right_clean/total
    args_dict["empty masked questions"]=empty_masked_q
    args_dict["questions not complete"]=qs_not_complete   
    args_dict["failed match questions"] = fail_match_qs
    args_dict["failed clean questions"] = fail_clean_qs
    args_dict["token_metrics"]= {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy_token
    },
    args_dict["confusion_matrix"]= {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    },
    args_dict["total"]=total
    print("-------------")
    print("!!! right match: ",right_match, "    right clean: ",right_clean, "   total: ", total, "    empty masked questions:", empty_masked_q)
    print("!!! accuracy match: ", right_match/total, "!!! accuracy clean: ", right_clean/total, "!!! Qs not complete: ", qs_not_complete_count)
    print("-------------")
    

    ### Save args
    #json_path = os.path.join(folder_name,str(max_gen_len)+"_args.json")
    basej_name = "_args"
    json_path=os.path.join(folder_name,f"1_{basej_name}.json")
    # if os.path.isfile(out_path):
    #     assert False
    counter = 1
    # Find next available filename
    while os.path.isfile(json_path):
        json_path = os.path.join(folder_name, f"{counter}_{basej_name}.json")
        counter += 1
    with open(json_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
    print(f"Arg MASKING stage will be saved to: {json_path}")
    print("------------------------------ END MASKING ------------------------------")

    
    fail_success_split_qs = {"fail_match_qs": fail_match_qs, "fail_clean_qs": fail_clean_qs, "success_match_qs": success_match_qs, "success_clean_qs": success_clean_qs}

    return args_dict, masked_data, fail_success_split_qs, masked_data_path #masked_data is list of dicts, there might be a more efficient way to format




# for each row in dataset, classify and add column ["send_to_cloud"] = 0 or 1
def classify(fail_success_split_qs, dataset="CyberNERQA", masked_data_path='', train_split=0.5, fail_split='match'):
    
    # set before reassign
    args_dict = {}
    args_dict['dataset']=dataset
    args_dict['train_split']=train_split
    args_dict['fail_split']=fail_split
    
    # split based on clean or match (to balance classes)
    fail_qs = fail_success_split_qs[f"fail_{fail_split}_qs"]
    success_qs = fail_success_split_qs[f"success_{fail_split}_qs"]

    # train test splits
    train_fail_qs = fail_qs[:int(len(fail_qs)*train_split)]
    train_success_qs = success_qs[:int(len(success_qs)*train_split)]
    test_fail_qs = fail_qs[int(len(fail_qs)*train_split):]
    test_success_qs = success_qs[int(len(success_qs)*train_split):]
    
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
    
    failed_embs = semantic_model.encode(train_fail_qs, convert_to_tensor=True)
    success_embs = semantic_model.encode(train_success_qs, convert_to_tensor=True)

    def individual_classify(failed_embs, success_embs, question_text):
        """
        Classify whether question should go to cloud or edge.
        Returns: 0 for edge, 1 for cloud
        """
        if semantic_model is None:
            return 1  # Default to cloud if no model
        
        q_emb = semantic_model.encode(question_text, convert_to_tensor=True)
        
        sim_fail = util.cos_sim(q_emb, failed_embs).max()
        sim_success = util.cos_sim(q_emb, success_embs).max()

        return 0 if sim_fail > sim_success else 1

    masked_classified_data=[]
    cloud_data=[]
    edge_data=[]
    edge_indices=[]
    cloud_indices=[]
    tp, fp, fn, tn = 0, 0, 0, 0 
    dataset = CyberNERQA_dataset(tokenizer=None, data_path=masked_data_path, args={}, step='classify') #check if I need to pass more args

    # keep track for splitting edge cloud later
    test_start_idx = int(len(dataset)*train_split)

    
    for i in range(int(len(dataset)*train_split),len(dataset)):
        item = dict(dataset.data[i])
        gt = item[f'masked_{fail_split}']
        
        pred = individual_classify(failed_embs, success_embs, item["Question"])
        
        item['send_to_cloud'] = pred # see how to add to each row efficiently, maybe just for loop
        masked_classified_data.append(item)
        
        test_relative_idx = i - test_start_idx
        
        if pred == 0:
            edge_data.append(item)
            edge_indices.append(test_relative_idx)
        else:
            cloud_data.append(item)
            cloud_indices.append(test_relative_idx)

        if gt == 1 and pred == 1:
            tp += 1  # True Positive: correctly identified PII
        elif gt == 0 and pred == 1:
            fp += 1  # False Positive: incorrectly masked non-PII
        elif gt == 1 and pred == 0:
            fn += 1  # False Negative: missed PII
        elif gt == 0 and pred == 0:
            tn += 1  # True Negative: correctly kept non-PII

    
    
    #report performance
    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0, 4)
    accuracy = round((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0, 4)
    
    
    args_dict["token_metrics"]= {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    },
    args_dict["confusion_matrix"]= {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    },
    
    
    #### Save output dataset
    # save to a new json dataset with the new column? another dataloader used by stage 1 and 2
    folder_name = "data/CyberNERQA_masked_classified/"
    base_data_name = "CyberNERQA_masked_classified"
    counter = 1
    mc_data_path = os.path.join(folder_name,f"{base_data_name}_{counter}.json")
    while os.path.isfile(mc_data_path):
        counter += 1
        mc_data_path = os.path.join(folder_name, f"{base_data_name}_{counter}.json")
    print(f"Output masked json data will be saved to: {mc_data_path}")
    print("Set the Dataloader's data_path to this path ^")
    
    with open(mc_data_path, "w") as f:
        json.dump(masked_classified_data, f, indent=2)


    
    ### Save args
    #json_path = os.path.join(folder_name,str(max_gen_len)+"_args.json")
    basej_name = "args"
    json_path=os.path.join(folder_name,f"1_{basej_name}.json")
    # if os.path.isfile(out_path):
    #     assert False
    counter = 1
    # Find next available filename
    while os.path.isfile(json_path):
        json_path = os.path.join(folder_name, f"{counter}_{basej_name}.json")
        counter += 1
    with open(json_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)


    
    return cloud_data, edge_data, edge_indices, cloud_indices, mc_data_path





class CyberNERQA_dataset(Dataset): 
    # load from json file saved from stage 0 masking
    ### Stage 1 should use "Masked_Question" column, and stage 2 should use raw "Question" column! ###

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def __init__(self, tokenizer, data_path, args, step='mask'): #steps = mask, classify, guide (stage1), answer (stage2)
        self.args = args
        self.tokenizer = tokenizer
        self.step = step # can stage 2 be set to 0 for masking?
        self.data_path = data_path
        self.MASK_SYSTEM_PROMPT = (
            """Question contains masked entities: [CREDENTIALS]=keys/passwords, [IDENTITY]=names/IDs, [CONTACT]=email/phone, [DEVICE]=device IDs, [LOCATION]=IPs/addresses, [FINANCIAL]=account numbers, [DATE]=dates.

Mask all personal and identifying entities in the text below, using the appropriate tags. Return ONLY the masked question.

Example:
Question: User lisa.chen called from +1-555-0198 reporting her device IMEI-358273054098321 was stolen. What immediate actions should I take?
Masked Question: User [IDENTITY] called from [CONTACT] reporting her device [DEVICE] was stolen. What immediate actions should I take?"""
                                  )


        print(f"Loading CyberNERQA")

        # dataset = self.load_json(self.args.data_args['masked data path']) #self.args.data_args is a path to a json file. probably need to fix this
        if self.step == 'mask':
            dataset = self.load_json(self.data_path)['test'] #raw data
        elif self.step == 'classify':
            dataset = self.load_json(self.data_path) #masked data
        else:
            dataset = self.load_json(self.data_path) #masked classified data
            valid_len = len(dataset)
            
            if 'indices' in args and args['indices'] is not None:
                args['indices'] = [i for i in args['indices'] if i < valid_len] #maybe add print to check it's working right ***
                dataset = [dataset[int(i)] for i in args['indices']]
            
        self.data = self.format_data(dataset)

        
        # Handle precomputed big model outputs if stage2
        self.big_output_pre = []
        if self.step == 'answer':
            with jsonlines.open(args['big_output_path'], "r") as big_file:
                for i in big_file:
                    self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        if self.step=='answer':
            big_output_pre = self.big_output_pre[idx]
            return self.tokenize(ins, big_output_pre, self.tokenizer)
        elif self.step == 'classify':# Return raw text, not tokens
            return {
                "Question": ins["Question"],
                "Masked Question": ins["Masked Question"],
                "masked_clean": ins["masked_clean"],
                "masked_match": ins["masked_match"]
            }
        else: #guide, mask
            return self.tokenize(ins, None, self.tokenizer)



    def format_data(self, dataset):
        """
        Convert HF CyberNERQA entries into a common schema used by other datasets.
        """
        formatted = []
        for item in dataset:
            if self.step=='mask':
                formatted.append(item)
            elif self.step == 'classify':
                formatted.append(item)
            else: # guide or answer steps
                # Build a multiple-choice question string
                choices = f"Answer Choices: (A) {item['A']} (B) {item['B']} (C) {item['C']} (D) {item['D']}"
                
                if ((self.step=='guide') and (self.args['masking'] == 1) and (item['send_to_cloud']==1)):  #only provide masked questions to cloud when classified and set
                    question = item["Masked_Question"]
                else: # edge can take raw question
                    question = item["Question"]
                
                formatted.append({
                    "question": question.strip() + " " + choices,
                    "answer": item["Answer"],
                    "send_to_cloud": item["send_to_cloud"]
                })
            
                #print("step: ", self.step, "    Check if actually masked when on guidance stage: ", question)
                
        return formatted

    def tokenize(self, test_dict, big_output_pre, tokenizer):
        examplar = self.create_demo_text()

        if self.step=='answer':
            instruction = examplar + " Q: " + test_dict["question"] + "\nA: " + big_output_pre
            
            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                # max_length=1024,
                # padding="max_length"
                padding='max_length',
                max_length=512
            ) 
    
            
        elif self.step=='guide':
            instruction = examplar + " Q: " + test_dict["question"] + "\nA: "

            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                padding='max_length',
                max_length=1024
            )
            
        elif self.step == 'classify':
            # no tokenization needed
            return test_dict   

        elif self.step=='mask':
            instruction = f"{self.MASK_SYSTEM_PROMPT}\nQuestion:\n{test_dict['Question']}\nMasked Question:"
            
            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=512
            )
            
        return inputs


    def create_demo_text(self): #from SecQA loader
        direct_answer_trigger_for_fewshot = "The answer is"
        x, z, y = [], [], []
        
        # Example 1 
        x.append("I need to reset the firewall rules on device [DEVICE] after a security audit. What's the safest approach? Answer choices: (A) Reset to factory defaults and reconfigure from scratch with documented settings, (B) Copy rules from another device in the same network segment, (C) Disable the firewall temporarily and re-enable it, (D) Update the firmware first before making any changes.")
        z.append("") # explanation/reasoning column? Would need to create. Check accuracy first
        y.append("A")
    
        # Example 2
        x.append("How do I configure two-factor authentication for remote access to our production servers? Answer choices: (A) Use SMS-based codes as the primary 2FA method, (B) Set up email-based verification codes, (C) Enable biometric authentication only, (D) Implement TOTP authenticator apps or hardware security keys with backup codes.")
        z.append("")
        y.append("D")
    
        # Example 3
        x.append("Our employee [IDENTITY] (ID: [IDENTITY]) reported a phishing email at [CONTACT]. What should be our first response action? Answer choices: (A) Delete the email from Sarah's inbox immediately, (B) Isolate the email, scan for IOCs, and check if other users received similar messages, (C) Reply to the sender to gather more information, (D) Forward the email to all employees as a warning.")
        z.append("")
        y.append("B")
    
        # Example 4
        x.append("I'm seeing suspicious login attempts from IP [LOCATION] to account number [FINANCIAL]. What tool should I use to investigate? Answer choices: (A) A password manager to update credentials, (B) An antivirus scanner on the local machine, (C) SIEM logs and threat intelligence feeds to analyze the IP and login patterns, (D) A VPN client to mask the IP address.")
        z.append("")
        y.append("C")
    
        # Example 5
        x.append("My system is showing repeated failed SSH attempts. How can I harden SSH security? Answer choices: (A) Change the SSH port to a non-standard number only, (B) Disable SSH entirely and use RDP instead, (C) Implement key-based authentication, disable root login, use fail2ban, and restrict access by IP, (D) Increase the password complexity requirements.")
        z.append("")
        y.append("C")
    
        # Select number of few-shot examples
        index_list = list(range(self.args['few_shot']))
        
        demo_text = ""
        for i in index_list:
            demo_text += (
                "Q: " + x[i] + "\n"
                + "A: " + z[i] + " "
                + direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            )
    
        return demo_text
