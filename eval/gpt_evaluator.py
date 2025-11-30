#!/usr/bin/env python3
"""
GPT评估器
使用GPT-4o进行语义相似度评分的评估模块
"""

import re
import json
import time
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)


class GPTEvaluator:

    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 max_workers: int = None, max_retries: int = 3):
     
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install it with: pip install openai")
        
      
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = min(max_workers or 16, cpu_count())
        self.max_retries = max_retries
        
    
        self.scoring_criteria = (
            "You are an intelligent evaluator designed to evaluate the correctness and similarity of generative outputs for question-answer pairs. "
            "Your task is to compare the model prediction answer with the correct answer and determine if they match in meaning. Here's the scoring criteria:\n\n"
            "### Scoring Criteria:\n"
            "5 = Perfect match or Correct in meaning\n"
            "4 = Key information correct, minor flaws\n"
            "3 = Partially correct\n"
            "2 = Mostly wrong answer for key query, but some relevance\n"
            "1 = Completely wrong or nonsense sentences\n\n"
            "Your response must ONLY be the integer score (e.g., 4). DO NOT include any text or explanation."
        )
        
        logger.info(f"GPT: {model}")
    
    def score_single_sample(self, question: str, prediction: str, reference: str, 
                           metadata: Optional[Dict] = None, sample_id: str = "unknown") -> Dict:

        messages = [
            {
                "role": "system",
                "content": self.scoring_criteria
            },
            {
                "role": "user", 
                "content": (
                    f"Question: {question}\n"
                    f"Correct Answer: {reference}\n"
                    f"Predicted Answer: {prediction}\n\n"
                    "Please provide a score from 1 to 5 based on how well the predicted answer matches the correct answer."
                )
            }
        ]
        
        score = 1  
        error_msg = None
        
        for attempt in range(self.max_retries):
            try:
     
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,  
                    max_tokens=10,  
                    timeout=30
                )
                
                reply = response.choices[0].message.content.strip()
                
                # 提取分数
                match = re.search(r'[1-5]', reply)
                if match:
                    score = int(match.group())
                    break
                else:
                    logger.warning(f"eroor: {reply}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"sample {sample_id}  {attempt+1}  fail: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  
        
        result = {
            'sample_id': sample_id,
            'question': question,
            'prediction': prediction,
            'reference': reference,
            'gpt_score': score,
            'metadata': metadata or {},
            'error': error_msg
        }
        
        return result
    
    def batch_evaluate(self, samples: List[Dict], save_path: Optional[str] = None) -> List[Dict]:


        results = []
        

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
  
            future_to_sample = {}
            for i, sample in enumerate(samples):
                future = executor.submit(
                    self.score_single_sample,
                    sample.get('question', ''),
                    sample.get('prediction', ''),
                    sample.get('reference', sample.get('answer', '')),
                    sample.get('metadata', {}),
                    sample.get('sample_id', f'sample_{i}')
                )
                future_to_sample[future] = sample
            
 
            with tqdm(total=len(samples), desc="GPT") as pbar:
                for future in as_completed(future_to_sample):
                    try:
                        result = future.result()
                        results.append(result)
                        

                        if save_path:
                            with open(save_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        
                    except Exception as e:
                        logger.error(f"error: {e}")
                        
                    pbar.update(1)
        
        logger.info(f"GPT finish with {len(results)} samples")
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:

        if not results:
            return {"error": "no results"}
        

        scores = [r['gpt_score'] for r in results if 'gpt_score' in r]
        total_samples = len(results)
        
        analysis = {
            'total_samples': total_samples,
            'successful_evaluations': len(scores),
            'failed_evaluations': total_samples - len(scores),
            'success_rate': len(scores) / total_samples if total_samples > 0 else 0,
            'overall_score': {
                'raw_avg': round(sum(scores) / len(scores), 2) if scores else 0,
                'standardized': round(((sum(scores) / len(scores)) - 1) / 4 * 100, 2) if scores else 0,
                'distribution': {str(i): scores.count(i) for i in range(1, 6)}
            }
        }
        
   
        self._analyze_by_category(results, analysis)
        
        return analysis
    
    def _analyze_by_category(self, results: List[Dict], analysis: Dict):

        
   
        weather_scores = {}
        question_type_scores = {}
        difficulty_scores = {}
        sensor_impact_scores = {}
        
        for result in results:
            metadata = result.get('metadata', {})
            score = result.get('gpt_score')
            
            if score is None:
                continue
            

            weather = metadata.get('weather', 'unknown')
            if weather not in weather_scores:
                weather_scores[weather] = []
            weather_scores[weather].append(score)
            
            question_type = metadata.get('question_type', 'unknown')
            if question_type not in question_type_scores:
                question_type_scores[question_type] = []
            question_type_scores[question_type].append(score)
            

            difficulty = metadata.get('difficulty_level', 'unknown')
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            difficulty_scores[difficulty].append(score)
            
    
            sensor_impact = metadata.get('sensor_impact', 'unknown')
            if sensor_impact not in sensor_impact_scores:
                sensor_impact_scores[sensor_impact] = []
            sensor_impact_scores[sensor_impact].append(score)
        

        analysis['by_weather'] = self._calculate_category_stats(weather_scores)
        analysis['by_question_type'] = self._calculate_category_stats(question_type_scores)
        analysis['by_difficulty'] = self._calculate_category_stats(difficulty_scores)
        analysis['by_sensor_impact'] = self._calculate_category_stats(sensor_impact_scores)
    
    def _calculate_category_stats(self, category_scores: Dict[str, List[int]]) -> Dict:

        stats = {}
        
        for category, scores in category_scores.items():
            if scores:
                raw_avg = round(sum(scores) / len(scores), 2)
                standardized = round((raw_avg - 1) / 4 * 100, 2)
                
                stats[category] = {
                    'count': len(scores),
                    'raw_avg': raw_avg,
                    'standardized': standardized,
                    'distribution': {str(i): scores.count(i) for i in range(1, 6)}
                }
        
        return stats
    
    def save_detailed_results(self, results: List[Dict], analysis: Dict, output_dir: str):
       
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        

        results_file = os.path.join(output_dir, f"gpt_evaluation_results_{timestamp}.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        

        analysis_file = os.path.join(output_dir, f"gpt_evaluation_analysis_{timestamp}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"results saved: {output_dir}")
        logger.info(f"  results: {results_file}")
        logger.info(f"  analysis: {analysis_file}")
    
    def create_comparison_report(self, results_by_model: Dict[str, List[Dict]]) -> Dict:

        comparison = {
            'models': list(results_by_model.keys()),
            'overall_comparison': {},
            'by_category_comparison': {}
        }
        

        for model_name, results in results_by_model.items():
            analysis = self.analyze_results(results)
            comparison['overall_comparison'][model_name] = analysis['overall_score']
        

        categories = ['by_weather', 'by_question_type', 'by_difficulty', 'by_sensor_impact']
        
        for category in categories:
            comparison['by_category_comparison'][category] = {}
            

            all_subcategories = set()
            for results in results_by_model.values():
                analysis = self.analyze_results(results)
                if category in analysis:
                    all_subcategories.update(analysis[category].keys())
            

            for subcategory in all_subcategories:
                comparison['by_category_comparison'][category][subcategory] = {}
                
                for model_name, results in results_by_model.items():
                    analysis = self.analyze_results(results)
                    if category in analysis and subcategory in analysis[category]:
                        comparison['by_category_comparison'][category][subcategory][model_name] = \
                            analysis[category][subcategory]
        
        return comparison





if __name__ == "__main__":
