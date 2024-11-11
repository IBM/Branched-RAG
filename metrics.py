import os
import numpy as np
from typing import Optional, Dict, Union, Tuple, List
from prompts import prompts, generated
from llama_index.llms.ibm import WatsonxLLM


class LLMEvaluator:
    def __init__(self,
                 model_id: str,
                 url: str,
                 api_key: str,
                 project_id: str
                 ) -> None:
        self._MODEL = WatsonxLLM(model_id=model_id, url=url, apikey=api_key, project_id=project_id, max_new_tokens=4000)

    def _extract_score_and_reasons_from_response(
            self,
            system_prompt: str
    ) -> Union[float, Tuple[float, Dict]]:

        normalize = 10.0
        system_response = self._MODEL.complete(prompt=system_prompt).text


        if "Supporting Evidence" in system_response:
            score = 0.0
            supporting_evidence = ""
            for line in system_response.split('\n'):
                if "Score" in line:
                    score = generated.re_0_10_rating(line) / normalize
                if "Criteria" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        criteria = ":".join(parts[1:]).strip()
                if "Supporting Evidence" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        supporting_evidence = ":".join(parts[1:]).strip()
                        sub_str = system_response[system_response.find("Supporting Evidence") + len("Supporting Evidence")+1 : system_response.find("Score") - 1]
                        supporting_evidence = supporting_evidence.strip().strip('\n')
                        supporting_evidence += sub_str.strip().strip('\n')
            reasons = {
                'reason':
                    (
                        f"{'Criteria: ' + str(criteria) + ' ' if criteria else ''}\n"
                        f"{'Supporting Evidence: ' + str(supporting_evidence) if supporting_evidence else ''}"
                    )
            }
            return score, reasons
        else:
            return generated.re_0_10_rating(system_response.text) / normalize

    def context_relevance(
            self, question: str, context: str
    ) -> float:
        return generated.re_0_10_rating(
            str_val=self._MODEL.complete(
                prompt=prompts.QS_RELEVANCE.format(
                    question=question, statement=context
                )
            )[0].generated_text
        ) / 10.0

    def context_relevance_with_cot_reasons(
            self, question: str, context: str
    ) -> float:
        system_prompt = prompts.QS_RELEVANCE.format(question=question, statement=context)
        system_prompt = system_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        return self._extract_score_and_reasons_from_response(system_prompt=system_prompt)

    def answer_relevance(
            self, question: str, answer: str
    ) -> float:
        return generated.re_0_10_rating(
            str_val=self._MODEL.complete(
                prompt=prompts.PR_RELEVANCE.format(
                    prompt=question, response=answer
                )
            )[0].generated_text
        ) / 10.0

    def answer_relevance_with_cot_reasons(
            self, question: str, answer: str
    ) -> float:
        system_prompt = prompts.PR_RELEVANCE.format(prompt=question, response=answer)
        system_prompt = system_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )

        return self._extract_score_and_reasons_from_response(system_prompt=system_prompt)

    def faithfulness(
            self, question: str, answer: str, context: str
    ) -> float:
        system_prompt = prompts.LONG_FORM_ANSWER_PROMPT.format(question=question, answer=answer)

        system_response = self._MODEL.complete(prompt=system_prompt)
        system_response = system_response.text

        inputs = prompts.NLI_STATEMENTS_MESSAGE.format(context=context, statements=system_response)
        outputs = self._MODEL.complete(prompt=inputs).text

        score = ""
        final_answer = "Final verdict for each statement in order:"
        final_answer = final_answer.lower()

        for output in outputs.split('\n'):
            output = output.lower().strip()
            if final_answer in output:
                score = output[output.find(final_answer) + len(final_answer) :]

        if score == "":
            return np.nan

        verdict = score.strip().split(".")
        total = 0
        sum = 0
        for val in verdict:
            val = val.strip()
            if val == "yes":
                sum += 1
            elif val == "no":
                total += 1
            else:
                continue
        if sum == 0:
            return 0
        return sum / (sum + total)

