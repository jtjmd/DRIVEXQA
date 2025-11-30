#!/usr/bin/env python3

import nltk
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pycocoevalcap.cider.cider as cider_scorer

logger = logging.getLogger(__name__)


class EvaluationMetrics:

    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.smoothing_function = SmoothingFunction().method1
        
  
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(device)
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
            
  
        self.cider_scorer = cider_scorer.Cider()
        
        logger.info("Evaluation metrics initialized successfully")
    
    def compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
       
        bleu_scores = {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return bleu_scores
        
        for i in range(1, 5):
            scores = []
            for pred, ref in zip(predictions, references):
                try:
              
                    pred_tokens = nltk.word_tokenize(pred.lower())
                    ref_tokens = nltk.word_tokenize(ref.lower())
                    
                 
                    if i == 1:
                        weights = (1.0, 0, 0, 0)
                    elif i == 2:
                        weights = (0.5, 0.5, 0, 0)
                    elif i == 3:
                        weights = (0.33, 0.33, 0.33, 0)
                    else: 
                        weights = (0.25, 0.25, 0.25, 0.25)
                    
                    score = sentence_bleu(
                        [ref_tokens], 
                        pred_tokens, 
                        weights=weights,
                        smoothing_function=self.smoothing_function
                    )
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU-{i} for pair: {e}")
                    scores.append(0.0)
            
            bleu_scores[f"bleu_{i}"] = np.mean(scores) if scores else 0.0
        
        return bleu_scores
    
    def compute_rouge_l(self, predictions: List[str], references: List[str]) -> float:

        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return 0.0
        
        rouge_l_scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = self.rouge_scorer.score(ref, pred)
                rouge_l_scores.append(score['rougeL'].fmeasure)
            except Exception as e:
                logger.warning(f"Error computing ROUGE-L for pair: {e}")
                rouge_l_scores.append(0.0)
        
        return np.mean(rouge_l_scores) if rouge_l_scores else 0.0
    
    def compute_cider(self, predictions: List[str], references: List[str]) -> float:

        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return 0.0
        
        try:
   
            gts = {}
            res = {}
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                gts[i] = [ref]  
                res[i] = [pred]
            
          
            score, _ = self.cider_scorer.compute_score(gts, res)
            return float(score)
            
        except Exception as e:
            logger.warning(f"Error computing CIDEr: {e}")
            return 0.0
    
    def compute_meteor(self, predictions: List[str], references: List[str]) -> float:
 
        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return 0.0
        
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            try:
                # Tokenize
                pred_tokens = nltk.word_tokenize(pred.lower())
                ref_tokens = nltk.word_tokenize(ref.lower())
                
                # Compute METEOR
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing METEOR for pair: {e}")
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores) if meteor_scores else 0.0
    
    def compute_sentence_similarity(self, predictions: List[str], references: List[str]) -> float:
  
        if self.sentence_model is None:
            logger.warning("Sentence transformer not available, returning 0.0")
            return 0.0
        
        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return 0.0
        
        try:

            pred_embeddings = self.sentence_model.encode(predictions)
            ref_embeddings = self.sentence_model.encode(references)
            
 
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing sentence similarity: {e}")
            return 0.0
    
    def compute_length_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:

        if len(predictions) != len(references):
            logger.error(f"Length mismatch: predictions={len(predictions)}, references={len(references)}")
            return {
                "avg_prediction_length": 0.0,
                "avg_groundtruth_length": 0.0,
                "prediction_length_ratio": 0.0
            }
        
        try:
           
            pred_lengths = [len(nltk.word_tokenize(pred)) for pred in predictions]
            ref_lengths = [len(nltk.word_tokenize(ref)) for ref in references]
            
            avg_pred_length = np.mean(pred_lengths)
            avg_ref_length = np.mean(ref_lengths)
  
            length_ratio = avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0.0
            
            return {
                "avg_prediction_length": avg_pred_length,
                "avg_groundtruth_length": avg_ref_length,
                "prediction_length_ratio": length_ratio
            }
            
        except Exception as e:
            logger.warning(f"Error computing length metrics: {e}")
            return {
                "avg_prediction_length": 0.0,
                "avg_groundtruth_length": 0.0,
                "prediction_length_ratio": 0.0
            }
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:

        logger.info(f"Computing metrics for {len(predictions)} predictions and {len(references)} references")
        

        bleu_scores = self.compute_bleu_scores(predictions, references)
        rouge_l = self.compute_rouge_l(predictions, references)
        cider = self.compute_cider(predictions, references)
        meteor = self.compute_meteor(predictions, references)
        sentence_similarity = self.compute_sentence_similarity(predictions, references)
        length_metrics = self.compute_length_metrics(predictions, references)
 
        all_metrics = {
            **bleu_scores,
            "rouge_l": rouge_l,
            "cider": cider,
            "meteor": meteor,
            "sentence_similarity": sentence_similarity,
            **length_metrics
        }
        
        logger.info("Metrics computation completed")
        return all_metrics
    
    def save_metrics(self, metrics: Dict[str, float], save_path: str):

        try:

            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.floating):
                    json_metrics[key] = float(value)
                elif isinstance(value, np.integer):
                    json_metrics[key] = int(value)
                else:
                    json_metrics[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
            logger.info(f"Metrics saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def save_predictions(self, predictions: List[str], references: List[str], 
                        questions: List[str], save_path: str):

        try:
            data = []
            for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
                data.append({
                    "index": i,
                    "question": q,
                    "prediction": pred,
                    "reference": ref
                })
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Predictions saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
