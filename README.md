Key research quests and answers on them:

1) General discussion analysis
   - here we used LDA (Latent Dirichlet Allocation (LDA)) algorithm (Bayesian network which is a core tool for ML and AI), 
     generative statistical model, for modeling automatically extracted topics in text corpora, 
     since it is considered as one the best for conducting research topic identification (here: problem identification) 
     when it comes to natural language.
   - according to the results of the algorith applied, we can identify following key themes emerging from the conversation. 
     These are connected to health issues of women (to be more specific menopause) and identifies possible need/urge/ 
     connection to seek assistance from a doctor and the overall need of common/comprehensive awareness with the problem. 
2) Treatment Options
    - there are various options (models) to use when need to identify certain words and specific terminology. 
        In our case, we might use BioBert, ClinicalBert, BlueBERT transformer models and fine-tune it on our data or use Med7 (spacy) model. 
    - we use BioBert, since it may provide us with more flexibility when it comes to NER i medical texts.
    - in this case we are about to choose among BioBert model, that was previously pretrained on biomedical texts/data, 
     thus it would be more efficient to get treatment options and identify which ones are the best (some percentage counter/metrics, etc.). 
      - 'BioBERT largely outperforms BERT and previous state-of-the-art
        models in a variety of biomedical text mining tasks when pre-trained on biomedical corpora. While BERT obtains
        performance comparable to that of previous state-of-the-art models, BioBERT significantly outperforms them on the
        following three representative biomedical text mining tasks: biomedical named entity recognition (0.62% F1 score
        improvement), biomedical relation extraction (2.80% F1 score improvement) and biomedical question answering
        (12.24% MRR improvement).'
      - 'Long Short-Term Memory (LSTM) and Conditional Random Field (CRF) have greatly improved performance in biomedical named entity recognition (NER) over the last few years'
      - 'BioBERT is the first domain-specific BERT based model pretrained on biomedical corpora for 23 days on eight NVIDIA V100 GPUs.'
      - 'BioBERT basically has the same structure as BERT. We briefly discuss the recently proposed BERT, and then we describe in detail the pre-training and fine-tuning process of BioBERT.'
      - 'NLP models designed for general purpose language understanding often obtains poor performance in biomedical text mining tasks. In this work, we pre-train BioBERT on PubMed abstracts (PubMed) and PubMed Central full-text articles (PMC).'
      - 'BioBERT, which is a pre-trained language representation model for biomedical text mining. We showed that pre-training BERT on biomedical corpora is crucial in applying it to the biomedical domain. Requiring minimal task-specific architectural modification, BioBERT outperforms previous models on biomedical text mining tasks such as NER, RE and QA.'
    - The overall analysis approach consisted of:
      1) identifying frequency of terms withing key phrases in our db
      2) use of dmis-lab/biobert-v1.1 model for tokenization purposes which was needed to encode (apply encoding) 
            when it commes to data information retrival (in our case: what treatment options were discussed and which ones received most attention)
      3) here we can see that one of the most efficient treatment options were: .... as presented on our visualization
            
4) Sentiment Analysis
    - here were used tf-Idf (widely used statistical model in NLP) which measures how important a term is whithin a document(relative to a collection of documents). 

    - what factors were driving people positive sentiment: crucial support, arc (ARC menoupause and cancer survivorship programme), crucial support, cimprehencive, woman health
    - the main causes of negative sentiment: chronicle, age despair, fatigue, microbiome
    - notable patterns in sentiment across different topics: ?






Notes: 

- when fine-tuning a model on a small dataset can lead to overfitting (consider ways to avoid this)
- Fine-tuning can sometimes result in forgetting important features from the pre-trained task. 
This is known as catastrophic forgetting and can impact the model’s ability to generalize.
