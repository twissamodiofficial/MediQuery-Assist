from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time


class DocumentClassifier:
    
    LABELS = [
        "lab report",
        "prescription",
        "discharge summary",
        "progress note",
        "imaging report",
        "consultation note",
        "operative report",
        "immunization record"
    ]
    
    def __init__(
        self, 
        pages_per_group=2,
        min_confidence=0.35,
        max_workers=4,
        model_name="cross-encoder/nli-deberta-v3-small"
    ):
        self.pages_per_group = pages_per_group
        self.min_confidence = min_confidence
        self.max_workers = max_workers
        self.model_name = model_name
        self.classifier = None
        
        print(f"[Classifier] Loading {model_name}...")
        self._load_model()
    
    def _load_model(self):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )
        print(f"[Classifier] Ready (device: {'GPU' if device >= 0 else 'CPU'})")
    
    def classify_document(self, file_path):
        start_time = time.time()
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                return self._default_result()
            
            print(f"[Classifier] Analyzing {len(pages)} pages...")
            
            page_groups = self._create_page_groups(pages)
            print(f"[Classifier] Created {len(page_groups)} groups, classifying in parallel...")
            
            group_results = self._classify_groups_parallel(page_groups)
            page_map = self._build_page_map(group_results)
            
            all_types = [p['type'] for p in page_map.values()]
            type_counts = defaultdict(int)
            for t in all_types:
                type_counts[t] += 1
            primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            unique_types = sorted(set(all_types), 
                                 key=lambda t: type_counts[t], 
                                 reverse=True)
            
            result = {
                "primary_type": primary_type,
                "page_classifications": page_map,
                "all_types": unique_types,
                "processing_time": round(time.time() - start_time, 2),
                "total_pages": len(pages)
            }
            
            print(f"[Classifier] Done in {result['processing_time']}s - "
                  f"Primary: {primary_type}, Types found: {len(unique_types)}")
            
            return result
            
        except Exception as e:
            print(f"[Classifier] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._default_result()
    
    def _create_page_groups(self, pages):
        groups = []
        for i in range(0, len(pages), self.pages_per_group):
            group_pages = pages[i:i + self.pages_per_group]
            page_nums = list(range(i + 1, i + len(group_pages) + 1))
            
            text = " ".join([p.page_content for p in group_pages])
            
            if len(text) > 2000:
                text = text[:1000] + " ... " + text[-1000:]
            
            groups.append({
                'text': text,
                'page_numbers': page_nums
            })
        
        return groups
    
    def _classify_groups_parallel(self, groups):
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_group = {
                executor.submit(self._classify_single_group, group): group
                for group in groups
            }
            
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    result = future.result()
                    result['page_numbers'] = group['page_numbers']
                    results.append(result)
                except Exception as e:
                    print(f"[Classifier] Group classification failed: {e}")
        
        return results
    
    def _classify_single_group(self, group):
        text = group['text']
        
        if not text.strip():
            return {'type': 'other', 'confidence': 0.0, 'scores': {}}
        
        result = self.classifier(text, self.LABELS, multi_label=True)
        
        primary_type = result['labels'][0]
        primary_score = result['scores'][0]
        
        scores = {
            label: score 
            for label, score in zip(result['labels'], result['scores'])
        }
        
        return {
            'type': primary_type,
            'confidence': primary_score,
            'scores': scores
        }
    
    def _build_page_map(self, group_results):
        page_map = {}
        
        for group in group_results:
            page_nums = group.get('page_numbers', [])
            doc_type = group.get('type', 'other')
            confidence = group.get('confidence', 0.0)
            
            for page_num in page_nums:
                page_map[page_num] = {
                    'type': doc_type,
                    'confidence': round(confidence, 2)
                }
        
        return page_map
    
    def _default_result(self):
        return {
            "primary_type": "other",
            "page_classifications": {},
            "all_types": ["other"],
            "processing_time": 0.0,
            "total_pages": 0
        }
