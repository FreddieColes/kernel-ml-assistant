#!/usr/bin/env python3
import numpy as np
import json
import time

class KernelCodeEvaluator:
    #Main evaluator class for kernel code generation
    
    def __init__(self, quality_threshold=60, confidence_threshold=0.5):
        #Initialize evaluator with thresholds
        #quality_threshold: minimum score to consider code as good quality
        #confidence_threshold: minimum confidence to predict code as good
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold
    
    def evaluate_code_quality(self, code):
        #Evaluate single code sample and return quality score 0-100
        if not code or not isinstance(code, str):
            return 0
        
        lines = code.split('\n')
        
        #Syntax correctness
        syntax_score = (
            (code.count('{') == code.count('}')) * 10 +
            (code.count('(') == code.count(')')) * 10 +
            (';' in code) * 5
        )
        
        #Code style
        tab_lines = sum(1 for line in lines if line.startswith('\t'))
        space_lines = sum(1 for line in lines if line.startswith('    '))
        style_score = (
            (tab_lines > space_lines) * 15 +
            (any(len(line) < 100 for line in lines)) * 10
        )
        
        #Kernel-specific content
        kernel_words = ['struct', 'rq', 'unlikely', 'return', 'task_struct']
        kernel_score = (sum(word in code for word in kernel_words) / len(kernel_words)) * 25
        
        #Patch format
        patch_score = (
            ('--- a/' in code) * 10 +
            ('+++ b/' in code) * 10 +
            ('@@' in code) * 5
        )
        
        total_score = syntax_score + style_score + kernel_score + patch_score
        return min(100, total_score)  #Cap at 100
    
    def calculate_metrics(self, quality_scores, confidence_scores):
        #Calculate accuracy, precision, recall, F1 from quality and confidence scores
        
        #Ground truth: is code actually good quality
        actual_good = np.array([q >= self.quality_threshold for q in quality_scores])
        
        #Predictions: does model predict good quality
        predicted_good = np.array([c >= self.confidence_threshold for c in confidence_scores])
        
        #Calculate confusion matrix components
        true_positives = np.sum(actual_good & predicted_good)
        false_positives = np.sum(~actual_good & predicted_good)
        true_negatives = np.sum(~actual_good & ~predicted_good)
        false_negatives = np.sum(actual_good & ~predicted_good)
        
        #Calculate metrics
        total = len(quality_scores)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'total_samples': total
        }
    
    def evaluate_batch(self, codes, confidences):
        #Evaluate batch of code samples
        print(f"Evaluating {len(codes)} code samples")
        
        #Calculate quality scores
        quality_scores = []
        for i, code in enumerate(codes):
            score = self.evaluate_code_quality(code)
            quality_scores.append(score)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(codes)} samples")
        
        #Calculate metrics
        metrics = self.calculate_metrics(quality_scores, confidences)
        
        #Add additional stats
        metrics.update({
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'quality_threshold': self.quality_threshold,
            'confidence_threshold': self.confidence_threshold
        })
        
        return metrics
    
    def generate_test_data(self, num_samples=1000):
        #Generate test data using actual kernel code generator
        from kernel_generation_model_training import HierarchicalXGBoostGenerator
        
        print(f"Generating {num_samples} kernel code samples")
        
        #Initialize the generator
        generator = HierarchicalXGBoostGenerator()
        
        if not generator.initialize_database("config.json"):
            print("Error: Database connection failed")
            return None, None, None
        
        #Train the generator
        print("Training generator")
        generator.train_hierarchical_models(limit=10000, cv_folds=3)
        
        #Create diverse test issues
        base_issues = [
            "CFS scheduler shows high latency during context switches",
            "Memory leak detected in NUMA allocator", 
            "Spinlock contention in interrupt handling path",
            "Real-time scheduler misses deadlines under heavy load",
            "Page allocator fragmentation under pressure",
            "Mutex deadlock detected under stress testing",
            "Load balancing fails with NUMA topology",
            "OOM killer triggers too aggressively",
            "RCU grace periods too long affecting latency",
            "Context switch overhead too high in fair scheduler",
            "RT scheduler priority inversion detected",
            "Idle balancing causes performance regression",
            "Task migration overhead impacts latency",
            "Scheduler tick causes jitter in RT tasks",
            "NUMA balancing causes performance issues",
            "Memory compaction blocks critical paths",
            "Slab allocator shows poor scalability",
            "Memory pressure handling needs improvement",
            "Page reclaim causes latency spikes",
            "NUMA node imbalance affects allocation"
        ]
        
        modifiers = [
            "",
            " with high CPU utilization",
            " under memory pressure",
            " during stress testing", 
            " on NUMA systems",
            " with mixed workloads",
            " in virtualized environments",
            " under I/O pressure",
            " with real-time constraints",
            " during system startup"
        ]
        
        #Generate diverse test cases
        test_issues = []
        for i in range(num_samples):
            base = base_issues[i % len(base_issues)]
            mod = modifiers[i % len(modifiers)]
            test_issues.append(base + mod)
        
        #Generate code using the actual generator
        print("Generating code with generator")
        batch_size = 50
        all_codes = []
        all_confidences = []
        
        for i in range(0, len(test_issues), batch_size):
            batch = test_issues[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(test_issues) + batch_size - 1)//batch_size}")
            
            try:
                results = generator.generate_code_batch(batch, batch_size=batch_size)
                
                for result in results:
                    all_codes.append(result['code'])
                    all_confidences.append(result['confidence_score'])
                    
            except Exception as e:
                print(f"Warning: Batch failed: {e}")
                #Add fallback for failed batches
                for issue in batch:
                    all_codes.append(f"/* Error generating code for: {issue[:50]} */")
                    all_confidences.append(0.1)
        
        print(f"Generated {len(all_codes)} code samples")
        return all_codes, all_confidences, test_issues
    
    def save_results(self, metrics, filename="thesis_results.json"):
        #Save results in thesis-ready format
        
        thesis_results = {
            'evaluation_metrics': {
                'accuracy': round(metrics['accuracy'], 4),
                'precision': round(metrics['precision'], 4), 
                'recall': round(metrics['recall'], 4),
                'f1_score': round(metrics['f1_score'], 4)
            },
            'confusion_matrix': {
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'true_negatives': metrics['true_negatives'],
                'false_negatives': metrics['false_negatives']
            },
            'dataset_statistics': {
                'total_samples': metrics['total_samples'],
                'mean_quality_score': round(metrics['mean_quality'], 2),
                'mean_confidence_score': round(metrics['mean_confidence'], 4),
                'quality_threshold': metrics['quality_threshold'],
                'confidence_threshold': metrics['confidence_threshold']
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(thesis_results, f, indent=2)
        
        print(f"Results saved to {filename}")
        return thesis_results
    
    def print_results(self, metrics):
        #Print results in clean format for thesis
        print("\n" + "="*50)
        print("MASTER'S THESIS EVALUATION RESULTS")
        print("="*50)
        
        print("\nCore Metrics:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        print("\nDataset Info:")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Mean Quality:  {metrics['mean_quality']:.2f}/100")
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        
        print("\nThresholds Used:")
        print(f"Quality Threshold:    {metrics['quality_threshold']}")
        print(f"Confidence Threshold: {metrics['confidence_threshold']}")

def main():
    #Main evaluation function
    print("Kernel Code Generation Evaluation")
    print("="*40)
    
    #Get sample size from user
    try:
        num_samples = int(input("Enter number of samples to generate (default 1000): ") or "1000")
    except ValueError:
        num_samples = 1000
    
    #Initialize evaluator
    evaluator = KernelCodeEvaluator()
    
    #Generate test data
    print("\nStep 1: Generating test data")
    codes, confidences, issues = evaluator.generate_test_data(num_samples)
    
    if codes is None:
        print("Error: Failed to generate test data")
        return None
    
    #Run evaluation
    print("\nStep 2: Running evaluation")
    metrics = evaluator.evaluate_batch(codes, confidences)
    
    #Display results
    evaluator.print_results(metrics)
    
    #Save results
    print("\nStep 3: Saving results")
    thesis_data = evaluator.save_results(metrics)
    
    print("\nEvaluation complete! Results ready for thesis.")
    return metrics

if __name__ == "__main__":
    main()