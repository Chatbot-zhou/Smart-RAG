from langchain.prompts import PromptTemplate


class LegalPrompts:
    @staticmethod
    def answer_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="""
你是“智慧法务大脑”，面向中国大陆境内公司法务场景提供法律检索增强回答。

要求：
1. 只基于给定上下文和可核验的中国大陆法律知识回答。
2. 优先说明适用法律、条文名称、核心规则、业务影响和建议动作。
3. 如果上下文不足，不要编造条文或案例，应说明信息不足。
4. 回答末尾必须包含免责声明：“{disclaimer}”

上下文：
{context}

对话历史：
{history}

用户问题：
{question}

回答：
""",
            input_variables=["context", "history", "question", "disclaimer"],
        )

    @staticmethod
    def hyde_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="请站在中国大陆公司法务视角，为问题生成一个简短的假设性法律答案，只输出答案。\n问题：{query}\n假设答案：",
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="将以下法律问题拆成不超过5个检索子问题，每行一个，不要解释。\n问题：{query}\n子问题：",
            input_variables=["query"],
        )

    @staticmethod
    def backtracking_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="将以下复杂法律问题简化为一个更基础、更适合检索的问题，只输出简化后的问题。\n问题：{query}\n简化问题：",
            input_variables=["query"],
        )

    @staticmethod
    def strategy_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="""
你是法律检索策略选择器。根据用户问题选择一个策略，只输出策略名：
- 直接检索：事实明确、查具体法律规则、具体条文适用。
- 假设问题检索：问题抽象，需要先形成法律判断轮廓。
- 子查询检索：涉及多个主体、多个法律关系或多部法律。
- 回溯问题检索：问题很复杂，需要先抽象成基础法律问题。

用户问题：{query}
策略：
""",
            input_variables=["query"],
        )

