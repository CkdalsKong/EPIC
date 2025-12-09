import json
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

# Load FactCC model
model_path = 'manueldeprada/FactCC'
print("Loading FactCC model...")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Model loaded successfully!\n")

# Path to the rewritten.jsonl file
input_file = '../output_prefwiki/wiki/EPIC/0/rewritten.jsonl'

# Read and process the jsonl file
results = []
incorrect_items = []
correct_items = []

print("Processing rewritten.jsonl...")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total_count = 0
incorrect_count = 0
correct_count = 0

for idx, line in enumerate(tqdm(lines, desc="Verifying")):
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    
    try:
        data = json.loads(line)
        original = data.get('original', '')
        rewritten = data.get('rewritten', '')
        
        if not original or not rewritten:
            continue
        
        total_count += 1
        
        # Use FactCC to verify: original is the source, rewritten is the claim
        input_dict = tokenizer(
            original, 
            rewritten, 
            max_length=512, 
            padding='max_length', 
            truncation='only_first', 
            return_tensors='pt'
        )
        
        logits = model(**input_dict).logits
        pred = logits.argmax(dim=1)
        label = model.config.id2label[pred.item()]
        
        result_item = {
            'index': idx,
            'original': original,
            'rewritten': rewritten,
            'reason': data.get('reason', ''),
            'relevant_preference': data.get('relevant_preference', ''),
            'prediction': label
        }
        results.append(result_item)
        
        if label == 'INCORRECT':
            incorrect_count += 1
            incorrect_items.append(result_item)
        else:
            correct_count += 1
            correct_items.append(result_item)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing line {idx}: {e}")
        continue

# Print summary
print("\n" + "="*80)
print("HALLUCINATION VERIFICATION RESULTS")
print("="*80)
print(f"\nTotal items processed: {total_count}")
print(f"CORRECT (No hallucination): {correct_count} ({correct_count/total_count*100:.2f}%)")
print(f"INCORRECT (Hallucination detected): {incorrect_count} ({incorrect_count/total_count*100:.2f}%)")

# Print incorrect items details
if incorrect_items:
    print("\n" + "="*80)
    print("INCORRECT (HALLUCINATED) ITEMS:")
    print("="*80)
    for item in incorrect_items:
        print(f"\n[Index {item['index']}]")
        print(f"Original: {item['original'][:200]}..." if len(item['original']) > 200 else f"Original: {item['original']}")
        print(f"Rewritten: {item['rewritten'][:200]}..." if len(item['rewritten']) > 200 else f"Rewritten: {item['rewritten']}")
        print(f"Reason: {item['reason'][:150]}..." if len(item['reason']) > 150 else f"Reason: {item['reason']}")
        print("-"*40)

# Save detailed results to a file
output_file = '../output_prefwiki/wiki/EPIC/0/hallucination_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'summary': {
            'total': total_count,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'correct_rate': correct_count/total_count*100 if total_count > 0 else 0,
            'incorrect_rate': incorrect_count/total_count*100 if total_count > 0 else 0
        },
        'incorrect_items': incorrect_items,
        'correct_items': correct_items
    }, f, ensure_ascii=False, indent=2)

print(f"\nDetailed results saved to: {output_file}")

