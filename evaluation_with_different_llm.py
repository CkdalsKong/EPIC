#!/usr/bin/env python3
"""
다른 LLM을 사용한 평가 스크립트
로컬 vLLM 서버 사용
"""

import os
import json
import requests
import time
import csv
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union
import argparse

class EvaluationWithDifferentLLM:
    def __init__(self, 
                 vllm_base_url: str = "http://localhost:8011",
                 evaluation_model: str = "meta-llama/Llama-3.3-70B-Instruct",
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 timeout: int = 60,
                 retry_count: int = 3):
        """
        다른 LLM을 사용하여 생성된 문서를 평가하는 클래스
        
        Args:
            vllm_base_url: vLLM 서버 URL (로컬 서버)
            evaluation_model: 평가에 사용할 모델명
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            timeout: API 호출 타임아웃 (초)
            retry_count: 재시도 횟수
        """
        self.vllm_base_url = vllm_base_url.rstrip('/')
        self.evaluation_model = evaluation_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retry_count = retry_count
        
        # 평가 메트릭 파일 경로
        self.error_type_dir = "data/error_type"
        
        # 서버 연결 확인
        self.check_server_connection()
        
    
    def check_server_connection(self) -> bool:
        """vLLM 서버 연결 상태 확인"""
        try:
            # 서버 상태 확인
            health_url = f"{self.vllm_base_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ vLLM 서버 연결 성공: {self.vllm_base_url}")
                
                # 사용 가능한 모델 확인
                models_url = f"{self.vllm_base_url}/v1/models"
                models_response = requests.get(models_url, timeout=10)
                
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [model["id"] for model in models_data.get("data", [])]
                    print(f"📋 사용 가능한 모델: {available_models}")
                    
                    if self.evaluation_model not in available_models:
                        print(f"⚠️ 경고: {self.evaluation_model} 모델이 서버에 로드되지 않았습니다.")
                        print(f"   사용 가능한 모델 중 하나를 선택하세요: {available_models}")
                        return False
                    else:
                        print(f"✅ 평가 모델 확인됨: {self.evaluation_model}")
                        return True
                else:
                    print(f"⚠️ 모델 목록 조회 실패: {models_response.status_code}")
                    return True  # 모델 목록 조회는 실패해도 계속 진행
            else:
                print(f"❌ vLLM 서버 연결 실패: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ vLLM 서버에 연결할 수 없습니다: {self.vllm_base_url}")
            print("   서버가 실행 중인지 확인하세요.")
            return False
        except requests.exceptions.Timeout:
            print(f"❌ vLLM 서버 응답 시간 초과: {self.vllm_base_url}")
            return False
        except Exception as e:
            print(f"❌ 서버 연결 확인 중 오류: {str(e)}")
            return False
        
    def load_evaluation_prompts(self) -> Dict[str, str]:
        """평가 메트릭 프롬프트들을 로드"""
        file_dict = {
            "acknow": "check_acknowledge.txt",
            "violate": "check_violation.txt", 
            "hallucinate": "check_hallucination.txt",
            "helpful": "check_helpful.txt"
        }
        
        prompts = {}
        for metric_name, file_name in file_dict.items():
            file_path = os.path.join(self.error_type_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    prompts[metric_name] = f.read()
            except FileNotFoundError:
                print(f"⚠️ Warning: {file_path} not found")
                prompts[metric_name] = ""
                
        return prompts
    
    def call_vllm_api(self, messages: List[Dict[str, str]], system_prompt: str = "") -> Optional[str]:
        """vLLM API를 호출하여 응답 생성 (재시도 로직 포함)"""
        for attempt in range(self.retry_count):
            try:
                payload = {
                    "model": self.evaluation_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "seed": 0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False,
                    "dtype": "float32"
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                
                # Add extra_body only for OSS model
                if self.evaluation_model == "openai/gpt-oss-20b":
                    payload["extra_body"] = {"reasoning_effort": "low"}
                
                response = requests.post(
                    f"{self.vllm_base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 503:
                    print(f"⚠️ 서버 과부하 (503), 재시도 중... ({attempt + 1}/{self.retry_count})")
                    if attempt < self.retry_count - 1:
                        import time
                        time.sleep(2 ** attempt)  # 지수 백오프
                        continue
                elif response.status_code == 400:
                    # 토큰 길이 오류인지 확인
                    error_text = response.text.lower()
                    if "context length" in error_text or "token" in error_text:
                        print(f"⚠️ 컨텍스트 길이 초과, 메시지 단축 시도... ({attempt + 1}/{self.retry_count})")
                        # 메시지 내용을 단축
                        shortened_messages = self.shorten_messages(messages)
                        if shortened_messages != messages:
                            messages = shortened_messages
                            continue
                    else:
                        print(f"❌ API 호출 실패: {response.status_code} - {response.text}")
                        return None
                else:
                    print(f"❌ API 호출 실패: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ API 호출 타임아웃, 재시도 중... ({attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.ConnectionError:
                print(f"❌ 서버 연결 오류: {self.vllm_base_url}")
                return None
            except Exception as e:
                print(f"❌ vLLM API 호출 중 오류: {str(e)}")
                return None
        
        print(f"❌ 최대 재시도 횟수 초과 ({self.retry_count})")
        return None
    
    def parse_explanation_and_answer(self, response: str) -> Tuple[str, str]:
        """응답에서 설명과 답변을 파싱"""
        try:
            # XML 형식 파싱 시도
            if "<explanation>" in response and "<answer>" in response:
                explanation_start = response.find("<explanation>") + len("<explanation>")
                explanation_end = response.find("</explanation>")
                answer_start = response.find("<answer>") + len("<answer>")
                answer_end = response.find("</answer>")
                
                explanation = response[explanation_start:explanation_end].strip()
                answer = response[answer_start:answer_end].strip()
            else:
                # 일반 텍스트 파싱
                lines = response.strip().split('\n')
                explanation = ""
                answer = ""
                
                for line in lines:
                    if line.lower().startswith(('yes', 'no')):
                        answer = line.strip()
                    else:
                        explanation += line.strip() + " "
                
                explanation = explanation.strip()
                
            return explanation, answer
            
        except Exception as e:
            print(f"⚠️ 파싱 오류: {str(e)}")
            return "", response.strip()
    
    def parse_preference_and_answer(self, response: str) -> Tuple[str, str]:
        """preference 추출과 답변을 파싱"""
        try:
            if "<extract_preference>" in response and "<answer>" in response:
                pref_start = response.find("<extract_preference>") + len("<extract_preference>")
                pref_end = response.find("</extract_preference>")
                answer_start = response.find("<answer>") + len("<answer>")
                answer_end = response.find("</answer>")
                
                extract_pref = response[pref_start:pref_end].strip()
                answer = response[answer_start:answer_end].strip()
            else:
                extract_pref = ""
                answer = response.strip()
                
            return extract_pref, answer
            
        except Exception as e:
            print(f"⚠️ preference 파싱 오류: {str(e)}")
            return "", response.strip()
    
    def evaluate_single_metric(self, task: Dict[str, Any], metric: str, 
                             eval_prompt: str, system_prompt: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """단일 메트릭에 대한 평가 수행"""
        try:
            preference = task.get("preference", "")
            question = task.get("question", "")
            response = task.get("response_to_q", "")
            
            if not response:
                return None
            
            # 프롬프트 텍스트 준비
            eval_text = eval_prompt
            if metric == "acknow":
                eval_text = eval_text.replace("{end_generation}", response).replace("{question}", question)
            elif metric in ["violate", "helpful"]:
                eval_text = eval_text.replace("{preference}", preference).replace("{question}", question).replace("{end_generation}", response)
            elif metric == "hallucinate":
                # acknowledge 결과가 필요
                error_check = task.get("evaluation_error_analysis", {})
                if "acknow" not in error_check:
                    return None
                extracted_pref = error_check["acknow"].get("extract_pref", "")
                eval_text = eval_text.replace("{preference}", preference).replace("{assistant_restatement}", extracted_pref)
            
            # vLLM API 호출
            messages = [{"role": "user", "content": eval_text}]
            eval_response = self.call_vllm_api(messages, system_prompt)
            
            if not eval_response:
                return None
            
            # 결과 파싱
            result = {}
            if metric != "acknow":
                explanation, answer = self.parse_explanation_and_answer(eval_response)
                result["explanation"] = explanation
                result["answer"] = answer
            else:
                extract_preference, answer = self.parse_preference_and_answer(eval_response)
                result["answer"] = answer
                result["extract_pref"] = extract_preference
            
            return metric, result
            
        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {str(e)}")
            return None
        
    def evaluate_generation_file(self, generation_file: str, output_file: Optional[str] = None) -> str:
        """생성 파일에 대한 평가 실행"""
        print(f"🔍 평가 시작: {generation_file}")
        print(f"🌐 사용 서버: {self.vllm_base_url}")
        print(f"🤖 평가 모델: {self.evaluation_model}")
        
        # 생성 데이터 로드
        try:
            with open(generation_file, 'r', encoding='utf-8') as f:
                generation_data = json.load(f)
        except Exception as e:
            print(f"❌ 파일 로드 실패: {str(e)}")
            return ""
        
        print(f"✅ {len(generation_data)}개의 생성 결과 로드됨")
        
        # 평가 프롬프트 로드
        eval_prompts = self.load_evaluation_prompts()
        system_prompt = "You are a helpful assistant in evaluating an AI assistant's response. You should be fair and strict and follow the user's instruction."
        
        # 출력 파일 설정
        if output_file is None:
            base_name = os.path.splitext(generation_file)[0]
            output_file = f"{base_name}_evaluated.json"
        
        # 메트릭 순서 정의 (acknow가 먼저 와야 hallucinate가 작동)
        metric_order = ["acknow", "violate", "helpful", "hallucinate"]
        
        # 각 메트릭을 순차적으로 처리
        for metric in metric_order:
            if metric not in eval_prompts or not eval_prompts[metric]:
                print(f"⚠️ {metric} 프롬프트가 없어서 건너뛰기")
                continue
                
            print(f"🔄 {metric} 메트릭 평가 중...")
            
            # 현재 메트릭에 대해 병렬 처리
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                
                for task_id, task in enumerate(generation_data):
                    if "response_to_q" not in task:
                        continue
                    
                    # 이미 현재 메트릭이 평가된 항목 건너뛰기
                    if "evaluation_error_analysis" in task:
                        analysis = task["evaluation_error_analysis"]
                        if metric in analysis:
                            continue
                    
                    # 현재 메트릭에 대해 평가
                    future = executor.submit(
                        self.evaluate_single_metric,
                        task,
                        metric,
                        eval_prompts[metric],
                        system_prompt
                    )
                    futures.append((task_id, future))
                
                # 결과 수집
                for task_id, future in tqdm(futures, desc=f"{metric} 평가 중"):
                    result = future.result()
                    if result:
                        metric_name, error_check = result
                        if "evaluation_error_analysis" not in generation_data[task_id]:
                            generation_data[task_id]["evaluation_error_analysis"] = {}
                        generation_data[task_id]["evaluation_error_analysis"][metric_name] = error_check
                    else:
                        print(f"⚠️ {metric} 평가 실패: task_id {task_id}")
        
        # preference_following_accuracy 추가
        for task in generation_data:
            if "evaluation_error_analysis" in task:
                pfa = self.calculate_preference_following_accuracy(task["evaluation_error_analysis"])
                task["preference_following_accuracy"] = pfa
        
        # 결과 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generation_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 평가 결과 저장됨: {output_file}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {str(e)}")
        
        return output_file
    
    def analyze_evaluation_results(self, evaluated_file: str) -> Dict[str, Any]:
        """평가 결과 분석"""
        try:
            with open(evaluated_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 평가 결과 파일 로드 실패: {str(e)}")
            return {}
        
        stats: Dict[str, Any] = {
            "total_responses": len(data),
            "acknowledgement": 0,
            "hallucination": 0,
            "violation": 0,
            "error_unhelpful": 0,
            "error_inconsistent": 0,
            "hallucination_of_preference_violation": 0,
            "preference_unaware_violation": 0,
            "preference_adherence_accuracy": 0,
        }
        
        for entry in data:
            if "evaluation_error_analysis" not in entry:
                continue
            
            error_types = entry["evaluation_error_analysis"]
            is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
            is_hallucination = is_acknowledgement and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
            is_violation = "yes" in error_types.get("violate", {}).get("answer", "").lower()
            is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()
            
            is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
            is_hallucination_of_preference_violation = (
                is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
            )
            is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful
            
            preference_following_accuracy = not any([
                is_inconsistent, is_hallucination_of_preference_violation, 
                is_preference_unaware_violation, is_unhelpful
            ])
            
            stats["acknowledgement"] += is_acknowledgement
            stats["hallucination"] += is_hallucination
            stats["violation"] += is_violation
            stats["error_unhelpful"] += is_unhelpful
            stats["error_inconsistent"] += is_inconsistent
            stats["hallucination_of_preference_violation"] += is_hallucination_of_preference_violation
            stats["preference_unaware_violation"] += is_preference_unaware_violation
            stats["preference_adherence_accuracy"] += preference_following_accuracy
        
        # 백분율 계산
        total = stats["total_responses"]
        if total > 0:
            stats["preference_following_accuracy_percent"] = round((stats["preference_adherence_accuracy"] / total) * 100, 2)
            stats["acknowledgement_percent"] = round((stats["acknowledgement"] / total) * 100, 2)
            stats["violation_percent"] = round((stats["violation"] / total) * 100, 2)
            stats["unhelpful_percent"] = round((stats["error_unhelpful"] / total) * 100, 2)
        
        return stats
    
    def calculate_preference_following_accuracy(self, error_analysis: Dict[str, Any]) -> int:
        """preference_following_accuracy 계산 (0 or 1)"""
        if not error_analysis:
            return 0
        
        is_acknowledgement = "yes" in error_analysis.get("acknow", {}).get("answer", "").lower()
        is_hallucination = is_acknowledgement and "yes" in error_analysis.get("hallucinate", {}).get("answer", "").lower()
        is_violation = "yes" in error_analysis.get("violate", {}).get("answer", "").lower()
        is_unhelpful = "no" in error_analysis.get("helpful", {}).get("answer", "").lower()
        
        is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
        is_hallucination_of_preference_violation = (
            is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
        )
        is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful
        
        preference_following_accuracy = not any([
            is_inconsistent, 
            is_hallucination_of_preference_violation, 
            is_preference_unaware_violation, 
            is_unhelpful
        ])
        
        return 1 if preference_following_accuracy else 0

def evaluate_organized_files(organized_dir, evaluator, method_filter=None, dataset_filter=None, model_filter=None):
    """
    정리된 파일들을 순회하면서 평가 수행
    """
    from pathlib import Path
    
    organized_path = Path(organized_dir)
    if not organized_path.exists():
        print(f"❌ 정리된 폴더가 존재하지 않습니다: {organized_dir}")
        return
    
    # 전체 결과 저장용
    all_results = []
    
    # 각 메소드별로 순회
    for method_dir in organized_path.iterdir():
        if not method_dir.is_dir():
            continue
            
        method_name = method_dir.name
        
        # 메소드 필터링
        if method_filter and method_name != method_filter:
            continue
            
        print(f"\n🔍 메소드 '{method_name}' 평가 시작")
        print("="*60)
        
        method_results = []
        
        # 각 데이터셋별로 순회
        for dataset_dir in method_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            
            # 데이터셋 필터링
            if dataset_filter and dataset_name != dataset_filter:
                continue
                
            print(f"\n📊 데이터셋 '{dataset_name}' 처리 중...")
            
            # 데이터셋별 CSV 파일이 있는지 확인
            dataset_csv_file = dataset_dir / f"{dataset_name}.csv"
            if dataset_csv_file.exists():
                print(f"    ⏭️ 데이터셋 CSV 파일이 이미 존재합니다: {dataset_csv_file}")
                # 기존 CSV에서 결과를 읽어와서 사용
                existing_results = load_existing_dataset_results(dataset_csv_file, method_name, dataset_name)
                if existing_results:
                    dataset_results = existing_results
                    method_results.extend(dataset_results)
                    all_results.extend(dataset_results)
                    print(f"    ✅ 기존 결과 {len(existing_results)}개 로드됨")
                    continue
            
            dataset_results = []
            
            # 각 모델별로 순회
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                
                # 모델 필터링
                if model_filter and model_name != model_filter:
                    continue
                    
                print(f"  🤖 모델 '{model_name}' 처리 중...")
                
                # JSON 파일들 찾기
                json_files = list(model_dir.glob("*.json"))
                print(f"    📁 {len(json_files)}개의 JSON 파일 발견")
                
                # 각 JSON 파일에 대해 평가 수행
                for json_file in json_files:
                    # eval 파일이 이미 있는지 확인
                    eval_file = json_file.parent / f"{json_file.stem}_evaluated.json"
                    if eval_file.exists():
                        print(f"      ⏭️ 평가 파일이 이미 존재합니다: {eval_file.name}")
                        # 기존 평가 결과를 로드
                        try:
                            stats = evaluator.analyze_evaluation_results(str(eval_file))
                            result = {
                                'method': method_name,
                                'dataset': dataset_name,
                                'model': model_name,
                                'file': json_file.name,
                                'evaluated_file': str(eval_file),
                                'stats': stats
                            }
                            dataset_results.append(result)
                            method_results.append(result)
                            all_results.append(result)
                            print(f"        ✅ 기존 결과 로드됨 - 정확도: {stats.get('preference_following_accuracy_percent', 0)}%")
                        except Exception as e:
                            print(f"        ⚠️ 기존 결과 로드 실패: {str(e)}")
                            # 기존 파일이 손상된 경우 새로 평가
                            continue
                    else:
                        print(f"      🔄 평가 중: {json_file.name}")
                        
                        try:
                            # 평가 실행 (eval 파일은 gen 파일과 같은 곳에 생성)
                            evaluated_file = evaluator.evaluate_generation_file(str(json_file))
                            
                            if evaluated_file:
                                # 결과 분석
                                stats = evaluator.analyze_evaluation_results(evaluated_file)
                                
                                print(f"        ✅ 평가 완료 - 정확도: {stats.get('preference_following_accuracy_percent', 0)}%")
                                
                                # 결과 저장
                                result = {
                                    'method': method_name,
                                    'dataset': dataset_name,
                                    'model': model_name,
                                    'file': json_file.name,
                                    'evaluated_file': evaluated_file,
                                    'stats': stats
                                }
                                
                                dataset_results.append(result)
                                method_results.append(result)
                                all_results.append(result)
                                
                            else:
                                print(f"        ❌ 평가 실패: {json_file.name}")
                                
                        except Exception as e:
                            print(f"        ❌ 평가 오류: {str(e)}")
            
            # 데이터셋별 결과를 txt 파일로 저장
            if dataset_results:
                save_dataset_results(dataset_dir, dataset_name, dataset_results)
        
        # 메소드별 전체 결과 저장
        if method_results:
            save_method_results(method_dir, method_name, method_results)
    
    # 전체 결과 저장
    if all_results:
        save_overall_results(organized_path, all_results)

def load_existing_dataset_results(csv_file, method_name, dataset_name):
    """기존 데이터셋 CSV 파일에서 결과를 로드"""
    try:
        results = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV에서 통계 정보를 복원
                stats = {
                    'total_responses': int(row.get('total_responses', 0)),
                    'preference_following_accuracy_percent': float(row.get('preference_following_accuracy(%)', 0)),
                    'acknowledgement_percent': float(row.get('acknowledgement(%)', 0)),
                    'violation_percent': float(row.get('violation(%)', 0)),
                    'unhelpful_percent': float(row.get('unhelpful(%)', 0)),
                    'preference_adherence_accuracy': int(row.get('preference_adherence_accuracy', 0)),
                    'acknowledgement': int(float(row.get('acknowledgement(%)', 0)) * int(row.get('total_responses', 0)) / 100) if int(row.get('total_responses', 0)) > 0 else 0,
                    'violation': int(float(row.get('violation(%)', 0)) * int(row.get('total_responses', 0)) / 100) if int(row.get('total_responses', 0)) > 0 else 0,
                    'error_unhelpful': int(float(row.get('unhelpful(%)', 0)) * int(row.get('total_responses', 0)) / 100) if int(row.get('total_responses', 0)) > 0 else 0,
                }
                
                result = {
                    'method': method_name,
                    'dataset': dataset_name,
                    'model': row.get('model', ''),
                    'file': row.get('file', ''),
                    'evaluated_file': '',  # CSV에서는 eval 파일 경로를 알 수 없음
                    'stats': stats
                }
                results.append(result)
        return results
    except Exception as e:
        print(f"    ❌ 기존 CSV 파일 로드 실패: {str(e)}")
        return []

def save_dataset_results(dataset_dir, dataset_name, dataset_results):
    """데이터셋별 결과를 txt 파일과 CSV 파일로 저장"""
    # persona_index별 정확도 추출
    persona_accuracies = {}
    
    for result in dataset_results:
        # 파일명에서 persona_index 추출 (예: gen_standard_flat_0.json -> 0)
        file_name = result['file']
        match = re.search(r'_(\d+)\.json$', file_name)
        if match:
            persona_index = int(match.group(1))
            accuracy = result['stats'].get('preference_following_accuracy_percent', 0)
            persona_accuracies[persona_index] = accuracy
    
    # CSV 파일로 저장
    csv_file = dataset_dir / f"{dataset_name}.csv"
    fieldnames = [
        "method", "dataset", "model", "file",
        "total_responses", "preference_following_accuracy(%)",
        "acknowledgement(%)", "violation(%)", "unhelpful(%)",
        "preference_adherence_accuracy"
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in dataset_results:
            stats = result['stats']
            writer.writerow({
                "method": result['method'],
                "dataset": result['dataset'],
                "model": result['model'],
                "file": result['file'],
                "total_responses": stats.get('total_responses', 0),
                "preference_following_accuracy(%)": stats.get('preference_following_accuracy_percent', 0),
                "acknowledgement(%)": stats.get('acknowledgement_percent', 0),
                "violation(%)": stats.get('violation_percent', 0),
                "unhelpful(%)": stats.get('unhelpful_percent', 0),
                "preference_adherence_accuracy": stats.get('preference_adherence_accuracy', 0)
            })
    
    # txt 파일로 저장
    txt_file = dataset_dir / f"{dataset_name}.txt"
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("="*50 + "\n\n")
        
        # persona_index별 정확도
        f.write("Persona Index별 정확도:\n")
        f.write("-" * 30 + "\n")
        for persona_idx in sorted(persona_accuracies.keys()):
            accuracy = persona_accuracies[persona_idx]
            f.write(f"Persona {persona_idx:2d}: {accuracy:6.2f}%\n")
        
        # 평균 정확도 계산
        if persona_accuracies:
            avg_accuracy = sum(persona_accuracies.values()) / len(persona_accuracies)
            f.write(f"\n평균 정확도: {avg_accuracy:.2f}%\n")
            f.write(f"총 Persona 수: {len(persona_accuracies)}\n")
        
        # 상세 통계
        f.write(f"\n상세 통계:\n")
        f.write("-" * 30 + "\n")
        total_responses = sum(r['stats'].get('total_responses', 0) for r in dataset_results)
        total_accuracy = sum(r['stats'].get('preference_adherence_accuracy', 0) for r in dataset_results)
        total_acknowledgement = sum(r['stats'].get('acknowledgement', 0) for r in dataset_results)
        total_violation = sum(r['stats'].get('violation', 0) for r in dataset_results)
        total_unhelpful = sum(r['stats'].get('error_unhelpful', 0) for r in dataset_results)
        
        f.write(f"총 응답 수: {total_responses}\n")
        f.write(f"전체 정확도: {(total_accuracy/total_responses*100):.2f}%\n" if total_responses > 0 else "전체 정확도: 0.00%\n")
        f.write(f"인정률: {(total_acknowledgement/total_responses*100):.2f}%\n" if total_responses > 0 else "인정률: 0.00%\n")
        f.write(f"위반률: {(total_violation/total_responses*100):.2f}%\n" if total_responses > 0 else "위반률: 0.00%\n")
        f.write(f"도움 안됨: {(total_unhelpful/total_responses*100):.2f}%\n" if total_responses > 0 else "도움 안됨: 0.00%\n")
    
    print(f"    💾 데이터셋 결과 저장: {txt_file}, {csv_file}")

def save_method_results(method_dir, method_name, method_results):
    """메소드별 결과를 CSV로 저장"""
    csv_file = method_dir / f"eval_{method_name}_summary.csv"
    
    fieldnames = [
        "method", "dataset", "model", "file",
        "total_responses", "preference_following_accuracy(%)",
        "acknowledgement(%)", "violation(%)", "unhelpful(%)",
        "preference_adherence_accuracy"
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in method_results:
            stats = result['stats']
            writer.writerow({
                "method": result['method'],
                "dataset": result['dataset'],
                "model": result['model'],
                "file": result['file'],
                "total_responses": stats.get('total_responses', 0),
                "preference_following_accuracy(%)": stats.get('preference_following_accuracy_percent', 0),
                "acknowledgement(%)": stats.get('acknowledgement_percent', 0),
                "violation(%)": stats.get('violation_percent', 0),
                "unhelpful(%)": stats.get('unhelpful_percent', 0),
                "preference_adherence_accuracy": stats.get('preference_adherence_accuracy', 0)
            })
    
    print(f"  💾 메소드별 결과 저장: {csv_file}")

def save_overall_results(organized_path, all_results):
    """전체 결과를 CSV로 저장"""
    overall_csv_file = organized_path / "eval_all_results.csv"
    
    fieldnames = [
        "method", "dataset", "model", "file",
        "total_responses", "preference_following_accuracy(%)",
        "acknowledgement(%)", "violation(%)", "unhelpful(%)",
        "preference_adherence_accuracy"
    ]
    
    with open(overall_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            stats = result['stats']
            writer.writerow({
                "method": result['method'],
                "dataset": result['dataset'],
                "model": result['model'],
                "file": result['file'],
                "total_responses": stats.get('total_responses', 0),
                "preference_following_accuracy(%)": stats.get('preference_following_accuracy_percent', 0),
                "acknowledgement(%)": stats.get('acknowledgement_percent', 0),
                "violation(%)": stats.get('violation_percent', 0),
                "unhelpful(%)": stats.get('unhelpful_percent', 0),
                "preference_adherence_accuracy": stats.get('preference_adherence_accuracy', 0)
            })
    
    print(f"\n💾 전체 결과 저장: {overall_csv_file}")
    
    # 전체 통계 계산
    total_responses = sum(r['stats'].get('total_responses', 0) for r in all_results)
    total_accuracy = sum(r['stats'].get('preference_adherence_accuracy', 0) for r in all_results)
    total_acknowledgement = sum(r['stats'].get('acknowledgement', 0) for r in all_results)
    total_violation = sum(r['stats'].get('violation', 0) for r in all_results)
    total_unhelpful = sum(r['stats'].get('error_unhelpful', 0) for r in all_results)
    
    overall_accuracy_percent = round((total_accuracy / total_responses) * 100, 2) if total_responses > 0 else 0
    overall_acknowledgement_percent = round((total_acknowledgement / total_responses) * 100, 2) if total_responses > 0 else 0
    overall_violation_percent = round((total_violation / total_responses) * 100, 2) if total_responses > 0 else 0
    overall_unhelpful_percent = round((total_unhelpful / total_responses) * 100, 2) if total_responses > 0 else 0
    
    print(f"\n📊 전체 평가 결과:")
    print(f"전체 응답 수: {total_responses}")
    print(f"전체 선호도 준수 정확도: {overall_accuracy_percent}%")
    print(f"전체 인정률: {overall_acknowledgement_percent}%")
    print(f"전체 위반률: {overall_violation_percent}%")
    print(f"전체 도움 안됨: {overall_unhelpful_percent}%")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='자동 평가 시스템')
    parser.add_argument('--organized_dir', type=str, default='organized_genfiles',
                       help='정리된 파일들이 있는 디렉토리')
    parser.add_argument('--method', type=str, default=None,
                       help='특정 메소드만 평가 (선택사항)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='특정 데이터셋만 평가 (선택사항)')
    parser.add_argument('--model', type=str, default=None,
                       help='특정 모델만 평가 (선택사항)')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8011', help='vLLM 서버 URL')
    parser.add_argument('--eval_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='평가 모델')
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = EvaluationWithDifferentLLM(
        vllm_base_url=args.vllm_url,
        evaluation_model=args.eval_model,
        max_tokens=512,
        temperature=0.0,
        timeout=60,
        retry_count=3
    )
    
    # 정리된 파일들 평가
    print("🚀 정리된 파일들에 대한 평가 시작")
    print(f"📁 정리된 폴더: {args.organized_dir}")
    
    if args.method:
        print(f"🔍 메소드 필터: {args.method}")
    if args.dataset:
        print(f"📊 데이터셋 필터: {args.dataset}")
    if args.model:
        print(f"🤖 모델 필터: {args.model}")
    
    evaluate_organized_files(
        args.organized_dir, 
        evaluator, 
        method_filter=args.method,
        dataset_filter=args.dataset,
        model_filter=args.model
    )

if __name__ == "__main__":
    main() 