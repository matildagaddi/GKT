#classifier
from sentence_transformers import SentenceTransformer

# for each row in dataset, classify and add column ["send_to_cloud"] = 0 or 1
def classifier(dataset=CyberNERQA):
    failed_idxs = []
    failed_qs = []
    success_qs = 
    for qid in range(len(train_data):
        data_row = dataset.data[qid]
        if data_row['clean'] == False:
            failed_idxs.append(qid)
            failed_qs.append(data_row['Question'])
        else:
            success_qs.append(data_row['Question'])
                             

    # split embeddings
    failed_embs = emb[list(failed_idxs)]
    success_embs = emb[[i for i in range(len(questions)) if i not in failed_idxs]]

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    def individual_classify(failed_embs, success_embs, question_text):
        """
        Classify whether question should go to cloud or edge.
        Returns: 0 for edge, 1 for cloud
        """
        if self.semantic_model is None:
            return 1  # Default to cloud if no model
        
        q_emb = semantic_model.encode(question_text, convert_to_tensor=True)
        
        sim_fail = util.cos_sim(q_emb, failed_embs).max()
        sim_success = util.cos_sim(q_emb, success_embs).max()

        return 0 if sim_fail > sim_success else 1
        
    
    classifications = [individual_classify(q["Question"]) for q in test_data]

    test_data['send_to_cloud'] = classifications # see how to add to each row efficiently, maybe just for loop