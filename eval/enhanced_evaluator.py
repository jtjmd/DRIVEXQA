#!/usr/bin/env python3


import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


from evaluation_metrics import EvaluationMetrics
from gpt_evaluator import GPTEvaluator
from metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class EnhancedEvaluator:

    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 original_qa_dir: Optional[str] = None,
                 enable_gpt_eval: bool = True,
                 gpt_max_workers: int = 8):

        self.metrics_calculator = EvaluationMetrics()
        

        self.metadata_extractor = MetadataExtractor()
        self.original_qa_dir = Path(original_qa_dir) if original_qa_dir else None
        

        self.gpt_evaluator = None
        self.enable_gpt_eval = enable_gpt_eval
        
        if enable_gpt_eval and openai_api_key:
            try:
                self.gpt_evaluator = GPTEvaluator(
                    api_key=openai_api_key,
                    max_workers=gpt_max_workers
                )
                logger.info("GPT initialized successfully")
            except Exception as e:
                logger.warning(f"GPT initialization failed: {e}")
                self.enable_gpt_eval = False
        elif enable_gpt_eval:
            logger.warning("No OpenAI API key provided, disabling GPT evaluation")
            self.enable_gpt_eval = False
        
        logger.info(f"Enhanced evaluator initialized, GPT evaluation: {'enabled' if self.enable_gpt_eval else 'disabled'}")
    
    def evaluate_predictions(self, predictions: List[str], references: List[str],
                           questions: List[str], metadata_samples: Optional[List[Dict]] = None) -> Dict:

        logger.info(f"Starting comprehensive evaluation of {len(predictions)} samples...")
        

        logger.info("Calculating traditional evaluation metrics...")
        traditional_metrics = self.metrics_calculator.compute_all_metrics(
            predictions=predictions,
            references=references
        )
        

        logger.info("Extracting/enhancing sample metadata...")
        if metadata_samples is None:

            metadata_samples = []
            for i, (q, p, r) in enumerate(zip(questions, predictions, references)):
                sample = {
                    'sample_id': f'sample_{i}',
                    'question': q,
                    'prediction': p,
                    'reference': r,
                    'scene_id': 'unknown',  
                    'timestamp': 'unknown'
                }
                metadata_samples.append(sample)
        

        enriched_samples = []
        for sample in metadata_samples:

            if 'question_type' in sample and sample.get('question_type') != 'unknown':

                metadata_dict = {
                    'weather': sample.get('weather', 'unknown'),
                    'question_type': sample.get('question_type', 'unknown'),
                    'sensor_impact': sample.get('sensor_impact', 'unknown')
                }
                enriched_sample = {**sample, 'metadata': metadata_dict}
                enriched_samples.append(enriched_sample)
            else:

               
                metadata = self.metadata_extractor.create_sample_metadata(sample, self.original_qa_dir)
               
                enriched_sample = {**sample, 'metadata': metadata}
                enriched_samples.append(enriched_sample)
        

        gpt_results = []
        gpt_analysis = {}
        
        if self.enable_gpt_eval and self.gpt_evaluator:
            logger.info("Executing GPT semantic similarity evaluation...")
            

            gpt_samples = []
            for sample in enriched_samples:
                gpt_sample = {
                    'sample_id': sample.get('sample_id', f'sample_{len(gpt_samples)}'),
                    'question': sample.get('question', ''),
                    'prediction': sample.get('prediction', ''),
                    'reference': sample.get('reference', sample.get('answer', '')),
                    'metadata': sample.get('metadata', {})
                }
                gpt_samples.append(gpt_sample)
            

            gpt_results = self.gpt_evaluator.batch_evaluate(gpt_samples)
            

            gpt_analysis = self.gpt_evaluator.analyze_results(gpt_results)
        

        logger.info("Generating comprehensive analysis report...")
        comprehensive_analysis = self._create_comprehensive_analysis(
            traditional_metrics, gpt_analysis, enriched_samples
        )
        

        category_analysis = self._create_category_analysis(enriched_samples, gpt_results)
        

        final_results = {
            'evaluation_summary': {
                'total_samples': len(predictions),
                'traditional_metrics_available': True,
                'gpt_evaluation_available': self.enable_gpt_eval,
                'metadata_enriched': True,
                'timestamp': datetime.now().isoformat()
            },
            'traditional_metrics': traditional_metrics,
            'gpt_analysis': gpt_analysis,
            'comprehensive_analysis': comprehensive_analysis,
            'category_analysis': category_analysis,
            'detailed_samples': enriched_samples,
            'gpt_results': gpt_results if gpt_results else None
        }
        
        logger.info("Comprehensive evaluation completed!")
        return final_results
    
    def _create_comprehensive_analysis(self, traditional_metrics: Dict, 
                                     gpt_analysis: Dict, 
                                     enriched_samples: List[Dict]) -> Dict:

        analysis = {
            'overall_performance': {},
            'correlation_analysis': {},
            'strength_weakness_analysis': {}
        }
        

        if traditional_metrics and gpt_analysis:
            analysis['overall_performance'] = {
                'bleu_4': traditional_metrics.get('bleu_4', 0),
                'rouge_l': traditional_metrics.get('rouge_l', 0),
                'bert_score_f1': traditional_metrics.get('bert_score_f1', 0),
                'gpt_score_standardized': gpt_analysis.get('overall_score', {}).get('standardized', 0)
            }
        

        analysis['data_distribution'] = self._analyze_data_distribution(enriched_samples)
        
        return analysis
    
    def _create_category_analysis(self, enriched_samples: List[Dict], 
                                gpt_results: List[Dict]) -> Dict:

        category_analysis = {
            'by_weather': {},
            'by_question_type': {},
            'by_sensor_impact': {}  
        }
        
        samples_by_category = {
            'weather': {},
            'question_type': {},
            'sensor_impact': {}  
        }
        
        for sample in enriched_samples:
            metadata = sample.get('metadata', {})
            
            for category_key, category_dict in samples_by_category.items():
                category_value = metadata.get(category_key, 'unknown')
                if category_value not in category_dict:
                    category_dict[category_value] = []
                category_dict[category_value].append(sample)
        
        for category_name, samples_dict in samples_by_category.items():
            analysis_key = f'by_{category_name}' if not category_name.endswith('_type') else f'by_{category_name}'
            if analysis_key not in category_analysis:
                analysis_key = f'by_{category_name.replace("_", "")}'
            
            category_analysis[analysis_key] = {}
            
            for category_value, samples in samples_dict.items():
                stats = {
                    'sample_count': len(samples),
                    'questions': [s.get('question', '') for s in samples[:3]],  
                }
                
                if gpt_results:
                    category_gpt_scores = []
                    for sample in samples:
                        sample_id = sample.get('sample_id', '')
                        for gpt_result in gpt_results:
                            if gpt_result.get('sample_id') == sample_id:
                                category_gpt_scores.append(gpt_result.get('gpt_score', 0))
                                break
                    
                    if category_gpt_scores:
                        stats['gpt_performance'] = {
                            'avg_score': sum(category_gpt_scores) / len(category_gpt_scores),
                            'score_distribution': {
                                str(i): category_gpt_scores.count(i) for i in range(1, 6)
                            }
                        }
                
                category_analysis[analysis_key][category_value] = stats
        
        return category_analysis
    
    def _analyze_data_distribution(self, enriched_samples: List[Dict]) -> Dict:
        distribution = {
            'weather_distribution': {},
            'question_type_distribution': {},
            'sensor_impact_distribution': {}  
        }
        
        for sample in enriched_samples:
            metadata = sample.get('metadata', {})
            
            weather = metadata.get('weather', 'unknown')
            question_type = metadata.get('question_type', 'unknown')
            sensor_impact = metadata.get('sensor_impact', 'normal_operation')
            
            distribution['weather_distribution'][weather] = \
                distribution['weather_distribution'].get(weather, 0) + 1
            distribution['question_type_distribution'][question_type] = \
                distribution['question_type_distribution'].get(question_type, 0) + 1
            distribution['sensor_impact_distribution'][sensor_impact] = \
                distribution['sensor_impact_distribution'].get(sensor_impact, 0) + 1
        
        return distribution
    
    def save_comprehensive_results(self, results: Dict, output_dir: str, 
                                 model_name: str = "model"):


        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        full_results_file = output_path / f"{model_name}_enhanced_evaluation_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        category_summary = {
            'model_name': model_name,
            'timestamp': timestamp,
            'category_analysis': results.get('category_analysis', {}),
            'comprehensive_analysis': results.get('comprehensive_analysis', {}),
            'evaluation_summary': results.get('evaluation_summary', {})
        }
        
        summary_file = output_path / f"{model_name}_category_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(category_summary, f, indent=2, ensure_ascii=False)
        
        if results.get('gpt_results'):
            gpt_file = output_path / f"{model_name}_gpt_results_{timestamp}.jsonl"
            with open(gpt_file, 'w', encoding='utf-8') as f:
                for gpt_result in results['gpt_results']:
                    f.write(json.dumps(gpt_result, ensure_ascii=False) + '\n')
        
        logger.info(f"Evaluation results saved to: {output_dir}")
        logger.info(f"  - complete results: {full_results_file}")
        logger.info(f"  - category summary: {summary_file}")
        if results.get('gpt_results'):
            logger.info(f"  - GPT results: {gpt_file}")
    
    def compare_models(self, results_by_model: Dict[str, Dict]) -> Dict:

        logger.info(f"Comparing performance of {len(results_by_model)} models...")
        
        comparison = {
            'models': list(results_by_model.keys()),
            'overall_comparison': {},
            'traditional_metrics_comparison': {},
            'gpt_scores_comparison': {},
            'category_performance_comparison': {}
        }
        
        for model_name, results in results_by_model.items():
            comparison['overall_comparison'][model_name] = {
                'traditional_metrics': results.get('traditional_metrics', {}),
                'gpt_overall_score': results.get('gpt_analysis', {}).get('overall_score', {}),
                'sample_count': results.get('evaluation_summary', {}).get('total_samples', 0)
            }
        
        comparison['traditional_metrics_comparison'] = self._compare_traditional_metrics(results_by_model)
        
        if any(results.get('gpt_analysis') for results in results_by_model.values()):
            comparison['gpt_scores_comparison'] = self._compare_gpt_scores(results_by_model)
        
        comparison['category_performance_comparison'] = self._compare_category_performance(results_by_model)
        
        logger.info("Model comparison analysis completed!")
        return comparison
    
    def _compare_traditional_metrics(self, results_by_model: Dict[str, Dict]) -> Dict:
        metrics_comparison = {}
        
        metric_names = ['bleu_1', 'bleu_4', 'rouge_l', 'bert_score_f1', 'exact_match']
        
        for metric in metric_names:
            metrics_comparison[metric] = {}
            for model_name, results in results_by_model.items():
                traditional_metrics = results.get('traditional_metrics', {})
                metrics_comparison[metric][model_name] = traditional_metrics.get(metric, 0)
        
        return metrics_comparison
    
    def _compare_gpt_scores(self, results_by_model: Dict[str, Dict]) -> Dict:
        gpt_comparison = {}
        
        for model_name, results in results_by_model.items():
            gpt_analysis = results.get('gpt_analysis', {})
            if gpt_analysis:
                overall_score = gpt_analysis.get('overall_score', {})
                gpt_comparison[model_name] = {
                    'raw_avg': overall_score.get('raw_avg', 0),
                    'standardized': overall_score.get('standardized', 0),
                    'distribution': overall_score.get('distribution', {}),
                    'by_category': {
                        category: gpt_analysis.get(category, {})
                        for category in ['by_weather', 'by_question_type', 'by_difficulty', 'by_sensor_impact']
                    }
                }
        
        return gpt_comparison
    
    def _compare_category_performance(self, results_by_model: Dict[str, Dict]) -> Dict:
        category_comparison = {}
        
        all_categories = set()
        for results in results_by_model.values():
            category_analysis = results.get('category_analysis', {})
            all_categories.update(category_analysis.keys())
        
        for category in all_categories:
            category_comparison[category] = {}
            
            all_subcategories = set()
            for results in results_by_model.values():
                category_data = results.get('category_analysis', {}).get(category, {})
                all_subcategories.update(category_data.keys())
            
            for subcategory in all_subcategories:
                category_comparison[category][subcategory] = {}
                
                for model_name, results in results_by_model.items():
                    category_data = results.get('category_analysis', {}).get(category, {})
                    subcategory_data = category_data.get(subcategory, {})
                    
                    category_comparison[category][subcategory][model_name] = {
                        'sample_count': subcategory_data.get('sample_count', 0),
                        'gpt_performance': subcategory_data.get('gpt_performance', {})
                    }
        
        return category_comparison




if __name__ == "__main__":
    test_enhanced_evaluator() 