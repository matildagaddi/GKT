#utils_data
import jsonlines
from torch.utils.data import Dataset
import json
import os
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

class Stage0_CyberNERQA_dataset(Dataset):
    # load dataset for stage 0 edge LLM to mask the "Question" column, 
    # and save new json dataset with extra "Masked_Question" column for stage 1 to use
    
    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def __init__(self, tokenizer, args, split="test", sample_idx=-1): #remove "stage2=False,"
        self.args = args
        self.tokenizer = tokenizer
        #self.stage2 = stage2 # can stage 2 be set to 0 for masking?
        self.sample_idx = sample_idx

        if split == "train":
            split_name = "dev"  # the dataset only has dev/val/test
        elif split == "validation":
            split_name = "val"
        else: #"test"
            split_name = split

        print(f"Loading CyberNERQA split: {split}")

        #load most recently generated masked dataset
        #data_path=os.path.join(self.args.data_path,self.args.dataset)[split_name] 
        data_path = 'data/CyberNERQA_raw/CyberNERQA.json' #hardcode for now
        self.data = self.load_json(data_path)[split] #probably just test for now
        self.MASK_SYSTEM_PROMPT = (
            """Question contains masked entities: [CREDENTIALS]=keys/passwords, [IDENTITY]=names/IDs, [CONTACT]=email/phone, [DEVICE]=device IDs, [LOCATION]=IPs/addresses, [FINANCIAL]=account numbers, [DATE]=dates.

Mask all personal and identifying entities in the text below, using the appropriate tags. Return ONLY the masked question.

Example:
Question: User lisa.chen called from +1-555-0198 reporting her device IMEI-358273054098321 was stolen. What immediate actions should I take?
Masked Question: User [IDENTITY] called from [CONTACT] reporting her device [DEVICE] was stolen. What immediate actions should I take?"""
                                  )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        # if self.stage2:
        #     big_output_pre = self.big_output_pre[idx]
        #     tokenized_full_data = self.tokenize(ins, big_output_pre, self.tokenizer)
        #else:
        tokenized_full_data = self.tokenize(ins, None, self.tokenizer)
        return tokenized_full_data

    def format_data(self, dataset):
        """
        Convert CyberNERQA entries into a common schema used by other datasets.
        """
        formatted = []
        for item in dataset:
            formatted.append({
                # "question": item["Question"].strip() + " " + choices,
                # "answer": item["Answer"]
                "Question": item["Question"].strip(),
                "PII-free Question": item["PII-free Question"], #correct
                "Entities": item["Entities"] #to check if these have at least been removed
        
            })
        return formatted

    def tokenize(self, test_dict, big_output_pre, tokenizer):
        #examplar = self.create_demo_text()

        # if self.stage2:
        #     instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: " + big_output_pre
        #else:
        #instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: " #use if using fewshots
        instruction = f"{self.MASK_SYSTEM_PROMPT}\nQuestion:\n{test_dict['Question']}\nMasked Question:"

        inputs = tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            # max_length=1024,
            # padding="max_length"
            padding='max_length',
            max_length=512
        )
        return inputs
    
    

class CyberNERQA_dataset(Dataset): 
    # load from json file saved from stage 0 masking
    ### Stage 1 should use "Masked_Question" column, and stage 2 should use raw "Question" column! ###

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def __init__(self, tokenizer, args, step='mask', split="test", sample_idx=-1): #steps = mask, (maybe classify), guide (stage1), answer (stage2)
        self.args = args
        self.tokenizer = tokenizer
        self.step = step # can stage 2 be set to 0 for masking?
        self.sample_idx = sample_idx
        self.MASK_SYSTEM_PROMPT = (
            """Question contains masked entities: [CREDENTIALS]=keys/passwords, [IDENTITY]=names/IDs, [CONTACT]=email/phone, [DEVICE]=device IDs, [LOCATION]=IPs/addresses, [FINANCIAL]=account numbers, [DATE]=dates.

Mask all personal and identifying entities in the text below, using the appropriate tags. Return ONLY the masked question.

Example:
Question: User lisa.chen called from +1-555-0198 reporting her device IMEI-358273054098321 was stolen. What immediate actions should I take?
Masked Question: User [IDENTITY] called from [CONTACT] reporting her device [DEVICE] was stolen. What immediate actions should I take?"""
                                  )
        #self.data_path = args.data_path

        # if split == "train":
        #     split_name = "dev"  # the dataset only has dev/val/test
        # elif split == "validation":
        #     split_name = "val"
        # else: #"test"
        #     split_name = split

        print(f"Loading CyberNERQA")

        # dataset = self.load_json(self.args.data_args['masked data path']) #self.args.data_args is a path to a json file. probably need to fix this
        if self.step == 'mask':
            dataset = self.load_json(self.args.data_args['raw_data_path'])['test']
        else:
            dataset = self.load_json(self.args.data_args['masked_classified_data_path'])
        self.data = self.format_data(dataset)

        
        # Handle precomputed big model outputs if stage2
        self.big_output_pre = []
        if sample_idx >= 0:
            #model_temp = f"{args.big_model_name.split('/')[-1]}2{args.small_model_name.split('/')[-1]}"
            folder_name = os.path.join(args.out_path, args.dataset, model_temp)
            big_out_path = os.path.join(folder_name, f"{split}_big.jsonlines")
            with jsonlines.open(big_out_path, "r") as big_file:
                for i in big_file:
                    self.big_output_pre.append(list(i.values())[0][sample_idx])
        elif self.step == 'guide':
            with jsonlines.open(args.big_output_path, "r") as big_file:
                for i in big_file:
                    self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        if self.step=='guide':
            big_output_pre = self.big_output_pre[idx]
            tokenized_full_data = self.tokenize(ins, big_output_pre, self.tokenizer)
        elif self.step=='answer':
            tokenized_full_data = self.tokenize(ins, None, self.tokenizer)
        elif self.step=='mask':
            tokenized_full_data = self.tokenize(ins, None, self.tokenizer)
        return tokenized_full_data


    def format_data(self, dataset):
        """
        Convert HF CyberNERQA entries into a common schema used by other datasets.
        """
        formatted = []
        for item in dataset:
            if self.step=='mask':
                formatted.append({
                    "Question": item["Question"].strip(),
                    "PII-free Question": item["PII-free Question"], #correct
                    "Entities": item["Entities"] #to check if these have at least been removed for "clean" acc
            
                })
            else: #not masking, on guide or answer steps
                # Build a multiple-choice question string
                choices = f"Answer Choices: (A) {item['A']} (B) {item['B']} (C) {item['C']} (D) {item['D']}"
                
                if ((self.step=='guide') and (self.args.masking == 1) and (item['send_to_cloud']==1)):  #only provide masked questions to cloud when classified and set
                    question = item["Masked_Question"]
                else: #edge can take raw question
                    question = item["Question"]
                
                formatted.append({
                    "question": question.strip() + " " + choices,
                    "answer": item["Answer"]
                })
            
                print("step: ", self.step, "    Check if actually masked when on guidance stage: ", question)
                
        return formatted

    def tokenize(self, test_dict, big_output_pre, tokenizer):
        examplar = self.create_demo_text()

        if self.step=='guide':
            instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: " + big_output_pre
            instruction = f"{self.MASK_SYSTEM_PROMPT}\nQuestion:\n{test_dict['Question']}\nMasked Question:"

            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                # max_length=1024,
                # padding="max_length"
                padding='max_length',
                max_length=512
            ) 
            
        elif self.step=='answer':
            instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: "

            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                padding='max_length',
                max_length=1024
            )

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
        index_list = list(range(self.args.few_shot))
        
        demo_text = ""
        for i in index_list:
            demo_text += (
                "Q: " + x[i] + "\n"
                + "A: " + z[i] + " "
                + direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            )
    
        return demo_text

        

class SecQA_dataset(Dataset):
    def __init__(self, tokenizer, args, stage2=False, split="test", sample_idx=-1):
        self.args = args
        self.tokenizer = tokenizer
        self.stage2 = stage2
        self.sample_idx = sample_idx

        # Load directly from Hugging Face
        if split == "train":
            split_name = "val"  # the dataset only has dev/val/test
        elif split == "validation":
            split_name = "val"
        else:
            split_name = split

        print(f"Loading SecQA split: {split_name}")
        dataset = load_dataset("zefang-liu/secqa", "secqa_v1", split=split_name)
        self.data = self.format_data(dataset)

        # Optionally truncate for debugging
        if hasattr(args, "debug_subset") and args.debug_subset:
            self.data = self.data[:len(self.data) // 10]

        # Handle precomputed big model outputs if stage2
        self.big_output_pre = []
        if sample_idx >= 0:
            model_temp = f"{args.big_model_name.split('/')[-1]}2{args.small_model_name.split('/')[-1]}"
            folder_name = os.path.join(args.out_path, args.dataset, model_temp)
            big_out_path = os.path.join(folder_name, f"{split}_big.jsonlines")
            with jsonlines.open(big_out_path, "r") as big_file:
                for i in big_file:
                    self.big_output_pre.append(list(i.values())[0][sample_idx])
        elif self.stage2:
            with jsonlines.open(args.big_output_path, "r") as big_file:
                for i in big_file:
                    self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        if self.stage2:
            big_output_pre = self.big_output_pre[idx]
            tokenized_full_data = self.tokenize(ins, big_output_pre, self.tokenizer)
        else:
            tokenized_full_data = self.tokenize(ins, None, self.tokenizer)
        return tokenized_full_data

    def format_data(self, dataset):
        """
        Convert HF SecQA entries into a common schema used by other datasets.
        """
        formatted = []
        for item in dataset:
            # Build a multiple-choice question string
            choices = f"Answer Choices: (A) {item['A']} (B) {item['B']} (C) {item['C']} (D) {item['D']}"
            formatted.append({
                "question": item["Question"].strip() + " " + choices,
                "answer": item["Answer"],
                "explanation": item["Explanation"]
            })
        return formatted

    def tokenize(self, test_dict, big_output_pre, tokenizer):
        examplar = self.create_demo_text()

        if "if_concise_prompt" in self.args and self.args.if_concise_prompt:
            system_prompt=self.args.if_concise_prompt
        else:
            system_prompt=""

        if self.stage2:
            instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: " + big_output_pre
        else:
            instruction = system_prompt + examplar + " Q: " + test_dict["question"] + "\nA: "

        inputs = tokenizer(
            instruction,
            return_tensors="pt",
            padding='max_length',
            max_length=1024
        )
        return inputs

    def create_demo_text(self):
        direct_answer_trigger_for_fewshot = "The answer is"
        x, z, y = [], [], []
        
        # Example 1
        x.append("What is a security concern associated with Software as a Service (SaaS)? Answer choices: (A) High costs associated with software licensing, (B) Consolidation of information with a single provider leading to potential data leaks, (C) The inability to customize software according to business needs, (D) The need for constant hardware upgrades.")
        z.append("A security concern with SaaS is the consolidation of information with a single provider. If the server running the SaaS is compromised, sensitive information, such as the Personally Identifiable Information (PII) of many users, may be at risk of being leaked.")
        y.append("B")
    
        # Example 2
        x.append("What is the purpose of implementing a Guest Wireless Network in a corporate environment? Answer choices: (A) To provide unrestricted access to company resources, (B) To replace the primary corporate wireless network, (C) To bypass network security protocols, (D) To offer a separate, secure network for visitors.")
        z.append("A Guest Wireless Network provides visitors with internet access while segregating them from the main corporate network, enhancing security by preventing unauthorized access to sensitive company resources.")
        y.append("D")
    
        # Example 3
        x.append("What is a typical indicator that an Intrusion Detection System (IDS) or Intrusion Prevention System (IPS) might identify as a network attack? Answer choices: (A) Anomalies or strange behaviors in network traffic, (B) Unauthorized software installation, (C) Frequent system reboots, (D) Regular updates to firewall rules.")
        z.append("IDS/IPS systems monitor network traffic and can identify network attacks by detecting anomalies, strange behaviors, or known exploit signatures in the traffic.")
        y.append("A")
    
        # Example 4
        x.append("What is the role of physical controls in a comprehensive security plan? Answer choices: (A) To solely manage digital threats such as viruses and malware, (B) To provide aesthetic enhancements to the security infrastructure, (C) To act as the primary defense against internal threats only, (D) To protect against physical access and breaches, complementing technical and administrative controls.")
        z.append("Physical controls, such as door locks and cameras, play a crucial role in protecting against physical access and breaches, and they complement technical and administrative controls to enhance overall security.")
        y.append("D")
    
        # Example 5
        x.append("Which feature should be enabled to hide the name of a wireless network, making it less visible to unauthorized users? Answer choices: (A) SSID Broadcast, (B) Enabling Firewall, (C) MAC Address Filtering, (D) Disabling DHCP")
        z.append("Disabling SSID Broadcast hides the network name from being openly broadcasted, which can help reduce the visibility of the network to unauthorized individuals.")
        y.append("A")
    
        # Select number of few-shot examples
        index_list = list(range(self.args.few_shot))
        
        demo_text = ""
        for i in index_list:
            demo_text += (
                "Q: " + x[i] + "\n"
                + "A: " + z[i] + " "
                + direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            )
    
        return demo_text
