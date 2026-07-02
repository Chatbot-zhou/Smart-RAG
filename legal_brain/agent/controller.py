from legal_brain.intent import intent_recognizer


class AgentDecisionController:
    """Four-layer Agent placeholder: intent, decision, execution, data."""

    def route(self, text: str) -> dict:
        intent, confidence = intent_recognizer.classify(text)
        route = {
            "legal_qa": "rag_answer",
            "regulation_query": "rag_answer",
            "risk_identification": "rag_answer",
            "contract_review": "contract_review_agent",
            "contract_drafting": "contract_drafting_agent",
        }[intent]
        return {"intent": intent, "confidence": confidence, "route": route}

    def contract_review_placeholder(self) -> dict:
        return {
            "status": "planned",
            "intent": "contract_review",
            "message": "合同审查 Agent 接口已预留，下一阶段接入规则生成、条款抽取、风险评级和 RAG 法律依据工具。",
            "required_next_steps": ["定义合同类型规则库", "接入条款解析器", "接入 RAG 法律依据工具", "实现审查报告模板"],
            "rag_tool_available": True,
        }

    def contract_drafting_placeholder(self) -> dict:
        return {
            "status": "planned",
            "intent": "contract_drafting",
            "message": "合同制作 Agent 接口已预留，下一阶段接入模板库、填槽校验、条款生成和人工确认流程。",
            "required_next_steps": ["建设合同模板库", "定义字段填槽 schema", "实现条款生成策略", "接入版本审阅流程"],
            "rag_tool_available": True,
        }


agent_controller = AgentDecisionController()
