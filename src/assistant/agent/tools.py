import re
from collections.abc import Callable
from dataclasses import asdict

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART = {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}


def _schema(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Поиск детали, замены, узла, ошибки или регламента по графу знаний.",
        {"question": {"type": "string"}},
    ),
    _schema(
        "summarize_topic",
        "Обзорные вопросы и статистика отказов по сводкам сообществ.",
        {"question": {"type": "string"}},
    ),
    _schema(
        "check_stock",
        "Проверить фактический остаток и discontinued в ERP на одном складе.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
    ),
    _schema("get_price", "Получить текущую цену детали в рублях и discontinued из ERP.", {"part_id": _PART}),
    _schema(
        "create_service_order",
        "Подготовить заказ; выполнение разрешает только подтверждение пользователя.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "pattern": "^(AR|BZ|GS)-[0-9]{3}$"},
        },
    ),
]


def validate_arguments(name: str, arguments: dict) -> str | None:
    if not isinstance(arguments, dict):
        return "Аргументы должны быть JSON-объектом"
    definitions = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}
    if name not in definitions:
        return "Неизвестный инструмент"
    definition = definitions[name]
    if set(arguments) != set(definition["required"]):
        return "Некорректный набор аргументов: " + ", ".join(definition["required"])
    for key, value in arguments.items():
        spec = definition["properties"][key]
        if spec["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            return f"{key}: нужна непустая строка"
        if spec["type"] == "integer" and (type(value) is not int or value < spec.get("minimum", 0)):
            return f"{key}: нужно положительное целое количество"
        if "enum" in spec and value not in spec["enum"]:
            return f"{key}: неизвестный склад"
        if "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
            return f"{key}: неверный формат кода"
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    def protect(name, action):
        def wrapped(**arguments):
            error = validate_arguments(name, arguments)
            if error:
                return {"error": error}
            try:
                return action(**arguments)
            except ErpNotFound as exc:
                return {"error": "Данных нет: " + str(exc)}
            except ErpRejected as exc:
                return {"error": "ERP отклонила запрос: " + str(exc)}

        return wrapped

    def knowledge(question):
        answer = local_search(question, kb, llm)
        result = asdict(answer)
        result["answer"] = result.pop("text")
        return result

    def topic(question):
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    return {
        "search_knowledge": protect("search_knowledge", knowledge),
        "summarize_topic": protect("summarize_topic", topic),
        "check_stock": protect("check_stock", erp.get_stock),
        "get_price": protect("get_price", erp.get_price),
        "create_service_order": protect("create_service_order", erp.create_order),
    }
