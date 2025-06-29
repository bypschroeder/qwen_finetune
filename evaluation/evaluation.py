import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv('evaluation/inference_results_q2.csv')

ground_truths = df['ground_truth'].astype(str).tolist()
preds_orig = df['pred_caption_orig'].astype(str).tolist()
preds_ft = df['pred_caption_ft'].astype(str).tolist()

smooth_fn = SmoothingFunction().method1

def compute_bleu(gts, preds):
    return [sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth_fn) for gt, pred in zip(gts, preds)]

def compute_meteor(gts, preds):
    return [
        meteor_score([word_tokenize(gt)], word_tokenize(pred))
        for gt, pred in zip(gts, preds)
    ]

def compute_rouge(gts, preds):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [scorer.score(gt, pred)['rougeL'].fmeasure for gt, pred in zip(gts, preds)]

def compute_bertscore(gts, preds):
    P, R, F1 = bert_score(preds, gts, lang='en', verbose=False)
    return F1.tolist()

def evaluate(gts, preds):
    bleu = compute_bleu(gts, preds)
    meteor = compute_meteor(gts, preds)
    rouge = compute_rouge(gts, preds)
    bert = compute_bertscore(gts, preds)
    return {
        'BLEU': sum(bleu) / len(bleu),
        'METEOR': sum(meteor) / len(meteor),
        'ROUGE-L': sum(rouge) / len(rouge),
        'BERTScore_F1': sum(bert) / len(bert)
    }

print('Evaluating original predictions:')
results_orig = evaluate(ground_truths, preds_orig)
for metric, value in results_orig.items():
    print(f'{metric}: {value:.4f}')

print('\nEvaluating fine-tuned predictions:')
results_ft = evaluate(ground_truths, preds_ft)
for metric, value in results_ft.items():
    print(f'{metric}: {value:.4f}')

# BLEU measures how closely your predicted captions match the reference captions in word sequences.
# METEOR evaluates your captions’ similarity to the references, accounting for synonyms and word order.
# ROUGE-L assesses the longest matching sequences of words between your captions and the references.
# BERTScore_F1 quantifies the semantic similarity between your captions and the references using deep language model embeddings.