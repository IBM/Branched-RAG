from metrics import LLMEvaluator

evaluator = LLMEvaluator(
    model_id="ibm/granite-13b-chat-v2",
    url="your_watson_url",
    api_key="your_api_key",
    project_id="your_project_id"
)

context_relevance_score, context_relevance_reasons = evaluator.context_relevance_with_cot_reasons(question, context)

answer_relevance_score, answer_relevance_reasons = evaluator.answer_relevance_with_cot_reasons(question, answer)

faithfulness_score = evaluator.faithfulness(question, answer, context)

result = {
    'question': question,
    'context': context,
    'answer': answer,
    'context_relevance': {
        'score': context_relevance_score,
        'reasons': context_relevance_reasons
    },
    'answer_relevance': {
        'score': answer_relevance_score,
        'reasons': answer_relevance_reasons
    },
    'faithfulness': faithfulness_score
}
