import json
import argparse
import re
from openai import OpenAI
from tqdm import tqdm

# Prompt template for hallucination verification
SYSTEM_PROMPT = """You are a factuality checker. Given an original document and its rewritten version, you must check whether the rewritten text introduces any hallucinated facts."""

USER_PROMPT_TEMPLATE = """Original chunk:

{source_chunk}

Rewritten chunk:

{rewritten_chunk}

For each sentence in the rewritten chunk, output a JSON list where each item has:

"sentence": the sentence text

"label": one of ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"] depending on whether the sentence is fully supported by the original chunk, contradicts it, or introduces new facts not present in the original. Do NOT rely on external world knowledge; only use the original chunk as evidence.

Output ONLY the JSON list, no other text."""


def parse_args():
    parser = argparse.ArgumentParser(description='Verify hallucination using LLM')
    parser.add_argument('--port', type=int, default=8000, help='vLLM server port')
    parser.add_argument('--input', type=str, default='../output_prefwiki/wiki/EPIC/0/rewritten.jsonl',
                        help='Input jsonl file path')
    parser.add_argument('--output', type=str, default='../output_prefwiki/wiki/EPIC/0/hallucination_results_llm.json',
                        help='Output json file path')
    return parser.parse_args()


def extract_json_from_response(response_text):
    """Extract JSON from LLM response, handling various formats."""
    # Try to find JSON array in the response
    try:
        # First, try direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code block
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(code_block_pattern, response_text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array pattern
    array_pattern = r'\[[\s\S]*\]'
    match = re.search(array_pattern, response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def verify_with_llm(client, model, original, rewritten):
    """Verify hallucination using LLM."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        source_chunk=original,
        rewritten_chunk=rewritten
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content
        sentence_analysis = extract_json_from_response(response_text)
        
        return {
            'success': True,
            'raw_response': response_text,
            'sentence_analysis': sentence_analysis
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'raw_response': None,
            'sentence_analysis': None
        }


def analyze_results(sentence_analysis):
    """Analyze sentence-level results to get summary."""
    if not sentence_analysis:
        return {'supported': 0, 'contradicted': 0, 'unsupported': 0, 'total': 0}
    
    counts = {'supported': 0, 'contradicted': 0, 'unsupported': 0, 'total': 0}
    
    for item in sentence_analysis:
        label = item.get('label', '').upper()
        counts['total'] += 1
        if label == 'SUPPORTED':
            counts['supported'] += 1
        elif label == 'CONTRADICTED':
            counts['contradicted'] += 1
        elif label == 'UNSUPPORTED':
            counts['unsupported'] += 1
    
    return counts


def main():
    args = parse_args()
    
    # Initialize OpenAI client for vLLM server
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="dummy"  # vLLM doesn't need real API key
    )
    
    model = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"Using vLLM server at port {args.port}")
    print(f"Model: {model}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}\n")
    
    # Read input file
    print("Reading input file...")
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each item
    results = []
    all_items = []
    
    # Summary counters
    total_sentences = 0
    total_supported = 0
    total_contradicted = 0
    total_unsupported = 0
    items_with_hallucination = []
    items_without_hallucination = []
    
    print("Processing items...")
    for idx, line in enumerate(tqdm(lines, desc="Verifying")):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            original = data.get('original', '')
            rewritten = data.get('rewritten', '')
            
            if not original or not rewritten:
                continue
            
            # Verify with LLM
            result = verify_with_llm(client, model, original, rewritten)
            
            # Analyze sentence-level results
            counts = analyze_results(result.get('sentence_analysis'))
            
            result_item = {
                'index': idx,
                'original': original,
                'rewritten': rewritten,
                'reason': data.get('reason', ''),
                'relevant_preference': data.get('relevant_preference', ''),
                'llm_response': result.get('raw_response'),
                'sentence_analysis': result.get('sentence_analysis'),
                'sentence_counts': counts,
                'has_hallucination': counts['contradicted'] > 0 or counts['unsupported'] > 0
            }
            
            all_items.append(result_item)
            
            # Update totals
            total_sentences += counts['total']
            total_supported += counts['supported']
            total_contradicted += counts['contradicted']
            total_unsupported += counts['unsupported']
            
            if result_item['has_hallucination']:
                items_with_hallucination.append(result_item)
            else:
                items_without_hallucination.append(result_item)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing line {idx}: {e}")
            continue
    
    # Print summary
    total_items = len(all_items)
    print("\n" + "="*80)
    print("HALLUCINATION VERIFICATION RESULTS (LLM-based)")
    print("="*80)
    print(f"\nTotal items processed: {total_items}")
    print(f"Items with potential hallucination: {len(items_with_hallucination)} ({len(items_with_hallucination)/total_items*100:.2f}%)")
    print(f"Items without hallucination: {len(items_without_hallucination)} ({len(items_without_hallucination)/total_items*100:.2f}%)")
    print(f"\nSentence-level analysis:")
    print(f"  Total sentences: {total_sentences}")
    print(f"  SUPPORTED: {total_supported} ({total_supported/total_sentences*100:.2f}%)" if total_sentences > 0 else "  SUPPORTED: 0")
    print(f"  CONTRADICTED: {total_contradicted} ({total_contradicted/total_sentences*100:.2f}%)" if total_sentences > 0 else "  CONTRADICTED: 0")
    print(f"  UNSUPPORTED: {total_unsupported} ({total_unsupported/total_sentences*100:.2f}%)" if total_sentences > 0 else "  UNSUPPORTED: 0")
    
    # Print items with hallucination
    if items_with_hallucination:
        print("\n" + "="*80)
        print("ITEMS WITH POTENTIAL HALLUCINATION:")
        print("="*80)
        for item in items_with_hallucination[:10]:  # Show first 10
            print(f"\n[Index {item['index']}]")
            print(f"Original: {item['original'][:200]}..." if len(item['original']) > 200 else f"Original: {item['original']}")
            print(f"Rewritten: {item['rewritten'][:200]}..." if len(item['rewritten']) > 200 else f"Rewritten: {item['rewritten']}")
            print(f"Counts: SUPPORTED={item['sentence_counts']['supported']}, CONTRADICTED={item['sentence_counts']['contradicted']}, UNSUPPORTED={item['sentence_counts']['unsupported']}")
            if item.get('sentence_analysis'):
                print("Sentence Analysis:")
                for sa in item['sentence_analysis']:
                    label = sa.get('label', 'N/A')
                    sentence = sa.get('sentence', '')[:100]
                    print(f"  [{label}] {sentence}...")
            print("-"*40)
        
        if len(items_with_hallucination) > 10:
            print(f"\n... and {len(items_with_hallucination) - 10} more items (see output file for full details)")
    
    # Save results
    output_data = {
        'summary': {
            'total_items': total_items,
            'items_with_hallucination': len(items_with_hallucination),
            'items_without_hallucination': len(items_without_hallucination),
            'hallucination_rate': len(items_with_hallucination)/total_items*100 if total_items > 0 else 0,
            'total_sentences': total_sentences,
            'supported_sentences': total_supported,
            'contradicted_sentences': total_contradicted,
            'unsupported_sentences': total_unsupported,
            'supported_rate': total_supported/total_sentences*100 if total_sentences > 0 else 0,
            'contradicted_rate': total_contradicted/total_sentences*100 if total_sentences > 0 else 0,
            'unsupported_rate': total_unsupported/total_sentences*100 if total_sentences > 0 else 0
        },
        'items_with_hallucination': items_with_hallucination,
        'items_without_hallucination': items_without_hallucination,
        'all_items': all_items
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

