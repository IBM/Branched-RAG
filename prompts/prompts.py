# NOTE: Try not to put anything new here.

from prompts import feedback

QS_RELEVANCE = feedback.QuestionStatementRelevance._prompt
PR_RELEVANCE = feedback.PromptResponseRelevance._prompt

LONG_FORM_ANSWER_PROMPT = feedback.LongFormAnswerPrompt._prompt
NLI_STATEMENTS_MESSAGE = feedback.NLIStatementsMessage._prompt

COT_REASONS_TEMPLATE = \
"""
Please answer with this template:

TEMPLATE: 
Criteria:<Provide the criteria for this evaluation>
Supporting Evidence:<Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
Score:<The score 0-10 based on the given criteria>
"""

