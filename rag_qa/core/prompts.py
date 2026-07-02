# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    # @staticmethod
    # def rag_prompt():
    #     # 创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个IT领域的智能问答助手，帮助用户回答问题，回答问题逻辑清晰、明确，语言风格专业、正式。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         问题: 『 {question} 』
    #         上下文:『 {context} 』
    #
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )

    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 基于提供的上下文（如果有）和你的知识回答问题。
           - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确的回答，避免无关信息。
           - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”

        **上下文**: {context}
        **对话历史**:
        {history}
        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )


# shift + { ，中文状态： 「『「『「『「『「『「『「
    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案。只保留答案，不要输出其他任何内容：  
            问题: 『 {query} 』  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询，最多不超过5个。不要输出其他任何内容：  
            查询: 『 {query} 』
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题，只保留简化后的问题，不要输出其他任何内容：  
            查询: 『 {query} 』
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )


if __name__ == '__main__':
    # rga_prompt = RAGPrompts.rag_prompt()
    # result = rga_prompt.format(context="黑马程序员", question="这个机构叫什么名称", phone="12345")
    # print(f'result-->{result}')
    # hyde = RAGPrompts.hyde_prompt()
    # result = hyde.format(query="你好吗")
    # print(result)

    rga_prompt = RAGPrompts.rag_prompt()
    # result = rga_prompt.format(context="黑马程序员成立于2006年，是一家优秀的互联网IT技术培训机构",
    #                            question="这个机构叫什么名称",
    #                            phone="12345")
    # print(f'result-->{result}')
    # subquery_prompt = RAGPrompts.subquery_prompt()
    # prompt_format = subquery_prompt.format(query='milvus和faiss的优缺点对比')
    # print(prompt_format)

    backtracking_prompt = RAGPrompts.backtracking_prompt()
    prompt_format = backtracking_prompt.format(query='我有一个100亿条数据的数据集，我想把它放到milvus中进行查询，可以吗？是否有可能出现性能不好的情况')
    print(prompt_format)