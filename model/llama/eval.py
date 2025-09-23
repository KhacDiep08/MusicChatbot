import json
import numpy as np
from sentence_transformers import CrossEncoder
from rag import RAGRetriever
from conversation import ConversationManager
import time

class ReRankerEvaluator:
    def __init__(self, eval_data_path, rag_db_path):
        self.eval_data = self.load_eval_data(eval_data_path)
        self.conv_manager = ConversationManager(rag_db_path=rag_db_path, use_rag=True)
        
        # Load Re-Ranker model 
        self.reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
        
        print("✅ Re-Ranker model loaded successfully")

    def load_eval_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_with_reranker(self, actual_response, expected_answer):
          
        pairs = [[actual_response, expected_answer]]
        
        scores = self.reranker.predict(pairs)
        
        return float(scores[0])

    def evaluate_retrieval_quality(self, question, expected_info):
        retrieved_results = self.conv_manager.rag.retrieve(question, top_k=5)
        
        retrieval_scores = []
        for result in retrieved_results:
            doc_text = str(result['song_info'])
            relevance_score = self.evaluate_with_reranker(doc_text, expected_info)
            retrieval_scores.append(relevance_score)
        
        return retrieval_scores

    def run_comprehensive_evaluation(self):
        results = []
        
        for i, test_case in enumerate(self.eval_data):
            print(f"🔍 Evaluating {i+1}/{len(self.eval_data)}: {test_case['question']}")
            
            start_time = time.time()
            response = self.conv_manager.generate_response(test_case['question'])
            response_time = time.time() - start_time
            
            reranker_score = self.evaluate_with_reranker(
                response, 
                test_case.get('expected_answer', test_case['question'])
            )
            
            expected_info = f"{test_case.get('expected_artist', '')} {test_case.get('expected_title', '')}"
            retrieval_scores = self.evaluate_retrieval_quality(test_case['question'], expected_info)
            
            result_entry = {
                'id': test_case['id'],
                'question': test_case['question'],
                'expected_answer': test_case.get('expected_answer', ''),
                'actual_response': response,
                'response_time_': round(response_time, 2),
                'reranker_score': round(reranker_score, 4),
                'retrieval_scores': [round(score, 4) for score in retrieval_scores],
                'max_retrieval_score': round(max(retrieval_scores), 4) if retrieval_scores else 0,
                'category': test_case['category']
            }
            
            results.append(result_entry)
            
            print(f"   ⏱️ Time: {response_time:.2f}s | Re-Ranker Score: {reranker_score:.4f}")
            print(f"   📊 Retrieval Scores: {[round(s, 3) for s in retrieval_scores[:3]]}")
            print(f"   💬 Response: {response[:100]}...\n")
        
        return results

    def calculate_metrics(self, results):

        reranker_scores = [r['reranker_score'] for r in results]
        retrieval_scores = [r['max_retrieval_score'] for r in results]
        response_times = [r['response_time_'] for r in results]
        
        return {
            'avg_reranker_score': round(np.mean(reranker_scores), 4),
            'avg_retrieval_score': round(np.mean(retrieval_scores), 4),
            'avg_response_time': round(np.mean(response_times), 2),
            'median_reranker_score': round(np.median(reranker_scores), 4),
            'success_rate_above_0.7': round(np.mean([1 if s > 0.7 else 0 for s in reranker_scores]), 4),
            'total_test_cases': len(results)
        }

    def analyze_failures(self, results):

        failures = [r for r in results if r['reranker_score'] < 0.5]
        
        failure_analysis = {
            'total_failures': len(failures),
            'failure_rate': round(len(failures) / len(results), 4),
            'common_failure_patterns': [],
            'low_retrieval_cases': [r for r in failures if r['max_retrieval_score'] < 0.3]
        }
        
        return failure_analysis

    def run(self, eval_queries=None):
        if eval_queries is None:
            return self.run_comprehensive_evaluation()
        results = []
        for q in eval_queries:
            resp = self.conv_manager.generate_response(q)
            score = self.evaluate_with_reranker(resp, q)
            results.append({"query": q, "response": resp, "score": score})
        return results

