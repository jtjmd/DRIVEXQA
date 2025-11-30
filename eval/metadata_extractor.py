#!/usr/bin/env python3

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MetadataExtractor:

    
  
    WEATHER_TYPES = {
        'sun': 'sunny',
        'fog': 'foggy', 
        'night': 'night',
        'rain': 'rainy'
    }
    

    QUESTION_CATEGORIES = {
        'global': 'Global',
        'local': 'Local', 
        'ego': 'Ego'
    }
    
 
    SENSOR_CONDITIONS = {
        'eventlowres': 'event_camera_degraded',
        'lidarjitter': 'lidar_jitter',
        'motionblur': 'motion_blur',
        'blur': 'motion_blur',  
        'overexposure': 'camera_overexposed',
        'underexposure': 'camera_underexposed',
        'camerafault': 'camera_overexposed',  
        'radarfault': 'lidar_jitter',  
        'noise': 'sensor_noise',  
        'normal': 'normal_operation'
    }
    
    def extract_scene_metadata(self, scene_id: str) -> Dict:
      
        metadata = {
            'weather': 'unknown'
        }
        
    
        pattern_with_condition = r'MAP_(\d+)_([a-zA-Z]+)_point(\d+)_([a-zA-Z_]+)'
        match = re.match(pattern_with_condition, scene_id)
        
        if match:
            weather_raw = match.group(2).lower()
            metadata['weather'] = self.WEATHER_TYPES.get(weather_raw, weather_raw)
        else:

            pattern = r'MAP_(\d+)_([a-zA-Z]+)_point(\d+)'
            match = re.match(pattern, scene_id)
            
            if match:
                weather_raw = match.group(2).lower()
                metadata['weather'] = self.WEATHER_TYPES.get(weather_raw, weather_raw)
                
        return metadata
    
    def extract_question_metadata(self, question: str, answer: str, 
                                original_qa_data: Optional[Dict] = None,
                                condition: Optional[str] = None) -> Dict:
      
        metadata = {
            'question_type': 'unknown',
            'sensor_impact': 'unknown'
        }
        


        if original_qa_data and 'qa_pairs' in original_qa_data:
            qa_pairs = original_qa_data['qa_pairs']
            
   
            found_match = False
            for level, categories in qa_pairs.items():
                if isinstance(categories, dict):
                    for category, qa_info in categories.items():
                        if isinstance(qa_info, dict) and 'question' in qa_info:
      
                            clean_question = qa_info['question'].lstrip(': ').strip().lower()
                            clean_input = question.lstrip(': ').strip().lower()
                            
           
                            match_found = False
                  
                            if clean_question == clean_input:
                                match_found = True
                      
                            elif len(clean_input) > 30 and clean_question in clean_input:
                                match_found = True
           
                            elif len(clean_input) > 10 and clean_input in clean_question:
                                match_found = True
                    
                            else:
       
                                stop_words = {'is', 'are', 'the', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'from', 'and', 'or', 'but', 'a', 'an', 'this', 'that', 'there', 'can', 'be', 'what', 'how', 'where', 'when', 'why', 'which', 'who'}
                                
                                question_words = [w for w in clean_question.split() if w not in stop_words and len(w) > 2]
                                input_words = [w for w in clean_input.split() if w not in stop_words and len(w) > 2]
                                
                                if len(question_words) > 0 and len(input_words) > 0:
                         
                                    matching_words = set(question_words) & set(input_words)
                                    match_ratio = len(matching_words) / max(len(question_words), len(input_words))
                                    
                                    if match_ratio >= 0.4:  
                                        match_found = True
     
                            
                            if match_found:
                   
                                metadata['question_type'] = category
                                found_match = True
                                break
                    
                    if found_match:
                        break
            
            if not found_match:
                print(f"unfound question tpyes for: {question[:50]}...")
        

        metadata['sensor_impact'] = self.extract_sensor_condition(condition)
        
        return metadata
    

    
    def extract_sensor_condition(self, condition: str) -> str:
        
        if not condition:
            return 'normal_operation'
        
        condition_lower = condition.lower()
  
        for sensor_type, status_name in self.SENSOR_CONDITIONS.items():
            if sensor_type in condition_lower:
                return status_name
 
        return 'normal_operation'
    

    
    def create_sample_metadata(self, sample: Dict, original_qa_dir: Optional[Path] = None) -> Dict:

        scene_id = sample.get('scene_id', '')
        scene_metadata = self.extract_scene_metadata(scene_id)
        

        condition = sample.get('condition', 'normal')
        

        metadata_dict = sample.get('metadata', {})
        conditions_array = metadata_dict.get('conditions', [])
        

        if conditions_array:
            for cond in conditions_array:
                cond_str = str(cond).lower()

                for sensor_key in self.SENSOR_CONDITIONS.keys():
                    if sensor_key in cond_str and sensor_key != 'normal':
                        condition = cond_str  
                        print(f"conditions: {cond_str}")
                        break
                if condition != 'normal':  
                    break
        

        original_qa_data = None
        if original_qa_dir:
            original_qa_data = self._load_original_qa_data(
                scene_id, 
                sample.get('timestamp', ''),
                original_qa_dir
            )

            if original_qa_data and 'condition' in original_qa_data and not condition:
                condition = original_qa_data['condition']
        

        question_metadata = self.extract_question_metadata(
            sample.get('question', ''),
            sample.get('answer', ''),
            original_qa_data,
            condition
        )
        

        metadata = {
            'weather': scene_metadata.get('weather', 'unknown'),
            'question_type': question_metadata.get('question_type', 'unknown'),
            'sensor_impact': question_metadata.get('sensor_impact', 'normal_operation')
        }
        
 
        logger.debug(f"- scene_id: {scene_id}, weather: {metadata['weather']}, question_type: {metadata['question_type']}, sensor_impact: {metadata['sensor_impact']}")
        
        return metadata
    
    def _load_original_qa_data(self, scene_id: str, timestamp: str, qa_dir: Path) -> Optional[Dict]:

        try:

            qa_file = qa_dir / scene_id / f"{timestamp}_qa.json"
            print(f"qa loaded: {qa_file}")

            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                return data
            

            scene_dir = qa_dir / scene_id
            if scene_dir.exists() and scene_dir.is_dir():

                qa_files = list(scene_dir.glob("*_qa.json"))

                
                if qa_files:
         
                    if len(qa_files) == 1:
                        qa_file = qa_files[0]

                    else:

                        qa_files.sort(key=lambda x: x.stem.split('_')[0] if x.stem.split('_')[0].isdigit() else '0', reverse=True)
                        qa_file = qa_files[0]

                    
                    with open(qa_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    return data
                    
        except Exception as e:

            logger.debug(f"can not load qa files {scene_id}/{timestamp}: {e}")

        return None
    
    def batch_extract_metadata(self, samples: List[Dict], 
                             original_qa_dir: Optional[Path] = None) -> List[Dict]:

        logger.info(f" start extracting {len(samples)} metadata..")
        
        enriched_samples = []
        for i, sample in enumerate(samples):
            try:
                metadata = self.create_sample_metadata(sample, original_qa_dir)
                enriched_sample = {**sample, 'metadata': metadata}
                enriched_samples.append(enriched_sample)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"precessed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                logger.warning(f"error: {e}")

                default_metadata = {'error': str(e)}
                enriched_sample = {**sample, 'metadata': default_metadata}
                enriched_samples.append(enriched_sample)
        
        logger.info(f"finished with {len(enriched_samples)} samples")
        return enriched_samples



if __name__ == "__main__":
